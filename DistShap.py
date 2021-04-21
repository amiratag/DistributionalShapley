
#______________________________________PEP8____________________________________
#_______________________________________________________________________
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import tensorflow as tf
import sys
from shap_utils import *
from scipy.stats import spearmanr
import shutil
from sklearn.base import clone
import time
import matplotlib.pyplot as plt
import itertools
import inspect
import _pickle as pkl
from sklearn.metrics import f1_score, roc_auc_score
import socket
import warnings
warnings.filterwarnings("ignore")



class DistShap(object):
    
    def __init__(self, X, y, X_test, y_test, num_test, X_tot=None, y_tot=None,
                 sources=None, 
                 sample_weight=None, directory=None, problem='classification',
                 model_family='logistic', metric='accuracy', seed=None,
                 overwrite=False,
                 **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assiging each point to its group.
                If None, evey points gets its individual value.
            samples_weights: Weight of train samples in the loss function
                (for models where weighted training method is enabled.)
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            overwrite: Delete existing data and start computations from 
                scratch
            **kwargs: Arguments of the model
        """
            
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_random_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if overwrite and os.path.exists(directory):
                tf.gfile.DeleteRecursively(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)  
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_test,
                                      X_tot, y_tot,
                                      sources, sample_weight)
        if len(set(self.y)) > 2:
            assert self.metric != 'f1', 'Invalid metric for multiclass!'
            assert self.metric != 'auc', 'Invalid metric for multiclass!'
        is_regression = (np.mean(self.y//1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        if self.is_regression:
            warnings.warn("Regression problem is no implemented.")
        self.model = return_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
        #if seed is None and self.directory is not None:
            #np.random.seed(int(self.experiment_number))
            #tf.random.set_random_seed(int(self.experiment_number))
            
            
    def _initialize_instance(self, X, y, X_test, y_test, num_test, 
                             X_tot=None, y_tot=None,
                             sources=None, sample_weight=None):
        """Loads or creates sets of data."""
        data_dir = os.path.join(self.directory, 'data.pkl')
        if not os.path.exists(data_dir):
            self._save_dataset(data_dir, X, y, X_test, y_test, num_test,
                               X_tot, y_tot, sources, sample_weight)  
        self._load_dataset(data_dir)
        loo_dir = os.path.join(self.directory, 'loo.npy')
        self.vals_loo = None
        if os.path.exists(loo_dir):
            self.vals_loo = np.load(loo_dir)
        self.experiment_number = self._find_experiment_number(self.directory)
        self._create_results_placeholder(
            self.experiment_number, len(self.X), len(self.sources))
    
    def _save_dataset(self, data_dir, X, y, X_test, y_test, num_test,
                      X_tot, y_tot, sources, sample_weight):
        '''Save the different sets of data if already does not exist.'''
        data_dic = {
            'X': X, 'y': y,
            'X_test': X_test[-num_test:], 'y_test': y_test[-num_test:],
            'X_heldout': X_test[:-num_test], 'y_heldout': y_test[:-num_test]
        }
        if sources is not None:
            data_dic['sources'] = sources
        if X_tot is not None:
            data_dic['X_tot'] = X_tot
            data_dic['y_tot'] = y_tot
        if sample_weight is not None:
            data_dic['sample_weight'] = sample_weight
            warnings.warn("Sample weight not implemented for G-Shapley")
        pkl.dump(data_dic, open(data_dir, 'wb'))
                      
    def _load_dataset(self, data_dir):
        '''Load the different sets of data if they already exist.'''
        
        data_dic = pkl.load(open(data_dir, 'rb'))
        self.X = data_dic['X'] 
        self.y = data_dic['y']
        self.X_test = data_dic['X_test']
        self.y_test = data_dic['y_test']
        self.X_heldout = data_dic['X_heldout']
        self.y_heldout = data_dic['y_heldout']
        if 'sources' in data_dic.keys() and data_dic['sources'] is not None:
            self.sources = data_dic['sources']
        else:
            self.sources = {i: np.array([i])
                            for i in range(len(self.X))}
        if 'X_tot' in data_dic.keys():
            self.X_tot = data_dic['X_tot']
            self.y_tot = data_dic['y_tot']
        else:
            self.X_tot = self.X
            self.y_tot = self.y
        if 'sample_weight' in data_dic.keys():
            self.sample_weight = data_dic['sample_weight']
        else:
            self.sample_weight = None   
          
    def _find_experiment_number(self, directory):
        
        '''Prevent conflict with parallel runs.'''
        if 'arthur' in socket.gethostname():
            flag = socket.gethostname()[-1]
        else:
            flag = '0'
        previous_results = os.listdir(directory)
        nmbrs = [int(name.split('.')[-2].split('_')[0][1:])
                 for name in previous_results
                 if '_result.pkl' in name and name[0] == flag]
        experiment_number = str(np.max(nmbrs) + 1) if len(nmbrs) else '0' 
        experiment_number = flag + experiment_number.zfill(5)
        print(experiment_number)
        return experiment_number
    
    def _create_results_placeholder(self, experiment_number, n_points, n_sources):
        '''Creates placeholder for results.'''
        self.results = {}
        self.results['mem_dist'] = np.zeros((0, n_points))
        self.results['mem_tmc'] = np.zeros((0, n_points))
        self.results['mem_g'] = np.zeros((0, n_points))
        self.results['idxs_dist'] = []
        self.results['idxs_tmc'] = []
        self.results['idxs_g'] = []
        self.save_results()
        
    def save_results(self, overwrite=False):
        """Saves results computed so far."""
        if self.directory is None:
            return
        results_dir = os.path.join(
            self.directory, 
            '{}_result.pkl'.format(self.experiment_number.zfill(6))
        )
        pkl.dump(self.results, open(results_dir, 'wb'))
    
    def restart_model(self):
        '''Restarts the model.'''
        try:
            self.model = clone(self.model)
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)
    
    def init_score(self, metric):
        """ Gives the value of an initial untrained model."""
        if metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float)/len(self.y_test)
            return np.max(hist)
        if metric == 'f1':
            rnd_f1s = []
            for _ in range(1000):
                rnd_y = np.random.permutation(self.y)
                rnd_f1s.append(f1_score(self.y_test, rnd_y))
            return np.mean(rnd_f1s)
        if metric == 'auc':
            return 0.5
        random_scores = []
        for _ in range(100):
            rnd_y = np.random.permutation(self.y)
            if self.sample_weight is None:
                self.model.fit(self.X, rnd_y)
            else:
                self.model.fit(self.X, rnd_y, 
                               sample_weight=self.sample_weight)
            random_scores.append(self.value(self.model, metric))
        return np.mean(random_scores)
        
    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data 
                different from test set.
            y: Labels, if valuation is performed on a data 
                different from test set.
            """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if inspect.isfunction(metric):
            return metric(model, X, y)
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))
        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)
        if metric == 'xe':
            return my_xe_score(model, X, y)
        raise ValueError('Invalid metric!')
        
    def _tol_mean_score(self):
        """Computes the average performance and its error using bagging."""
        scores = []
        self.restart_model()
        for _ in range(1):
            if self.sample_weight is None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(self.X, self.y,
                              sample_weight=self.sample_weight)
            for _ in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.value(
                    self.model, 
                    metric=self.metric,
                    X=self.X_test[bag_idxs], 
                    y=self.y_test[bag_idxs]
                ))
        self.tol = np.std(scores)
        self.mean_score = np.mean(scores)

    def run(self, save_every, err, tolerance=None, truncation=None, alpha=None,
            dist_run=False, tmc_run=False, loo_run=False,
            max_iters=None):
        """Calculates data sources(points) values.
        
        Args:
            save_every: Number of samples to to take at every iteration.
            err: stopping criteria (maximum deviation of value in the past 100 iterations).
            tolerance: Truncation tolerance. If None, it's computed.
            truncation: truncation for D-Shapley (if none will use data size).
            alpha: Weighted sampling parameter. If None, biased sampling is not performed.
            dist_run:  If True, computes and saves D-Shapley values.
            tmc_run:  If True, computes and saves TMC-Shapley values.
            loo_run: If True, computes and saves leave-one-out scores.
            max_iters: If not None, maximum number of iterations.
        """
        if loo_run:
            try:
                len(self.vals_loo)
            except:
                self.vals_loo = self._calculate_loo_vals(sources=self.sources)
                np.save(os.path.join(self.directory, 'loo.npy'), self.loo_vals)
            print('LOO values calculated!')
        iters = 0
        while dist_run or tmc_run:
            if dist_run:
                if error(self.results['mem_dist']) < err:
                    dist_run = False
                    print('Distributional Shapley has converged!')
                else:
                    self._dist_shap(
                        save_every, 
                        truncation=truncation, 
                        sources=self.sources,
                        alpha=alpha
                    )
                    self.vals_dist = np.mean(self.results['mem_dist'], 0)
            if tmc_run:
                if error(self.results['mem_tmc']) < err:
                    tmc_run = False
                    print('Data Shapley has converged!')
                else:
                    self._tmc_shap(
                        save_every, 
                        tolerance=tolerance, 
                        sources=self.sources
                    )
                    self.vals_tmc = np.mean(self.results['mem_tmc'], 0)
            if self.directory is not None:
                self.save_results()
            iters += 1
            if max_iters is not None and iters >= max_iters:
                print('Reached to maximum number of iterations!')
                break
        print('All methods have converged!')
        
    def _dist_shap(self, iterations, truncation, sources=None, alpha=None):
        """Runs Distribution-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model              
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} Dist_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.dist_iteration(
                truncation=truncation, 
                sources=sources,
                alpha=alpha
            )
            self.results['mem_dist'] = np.concatenate([
                self.results['mem_dist'], 
                np.reshape(marginals, (1,-1))
            ])
            self.results['idxs_dist'].append(idxs)
        
    def dist_iteration(self, truncation, sources=None, alpha=None):
        
        num_classes = len(set(self.y_test))
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        marginal_contribs = np.zeros(len(self.X))
        while True:
            k = np.random.choice(np.arange(1, truncation + 1))
            if alpha is None or np.random.random() < (1. / (k ** alpha)):
                break
        if k == 1:
            return marginal_contribs, []
        S = np.random.choice(len(self.X_tot), k - 1)
        X_init = self.X_tot[S]
        y_init = self.y_tot[S]
        self.restart_model()
        if len(set(y_init)) != num_classes and not self.is_regression:
            init_score = self.random_score
        else:
            try:
                self.model.fit(X_init, y_init)
                init_score = self.value(self.model, metric=self.metric)
            except:
                init_score = self.random_score
        time_init = time.time()
        for idx in range(len(sources.keys())):
            X_batch = np.concatenate([X_init, self.X[sources[idx]]])
            y_batch = np.concatenate([y_init, self.y[sources[idx]]])
            if len(set(y_batch)) != num_classes and not self.is_regression:
                continue
            self.restart_model()
            try:
                self.model.fit(X_batch, y_batch)
                score = self.value(self.model, metric=self.metric)
                marginal_contribs[sources[idx]] = score - init_score
                marginal_contribs[sources[idx]] /= len(sources[idx])
            except:
                continue
        return marginal_contribs, list(S)
        
    def _tmc_shap(self, iterations, tolerance=None, sources=None):
        """Runs TMC-Shapley algorithm.
        
        Args:
            iterations: Number of iterations to run.
            tolerance: Truncation tolerance ratio.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        model = self.model
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if tolerance is None:
            tolerance = self.tol         
        marginals, idxs = [], []
        for iteration in range(iterations):
            if 10*(iteration+1)/iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(
                    iteration + 1, iterations))
            marginals, idxs = self.tmc_iteration(
                tolerance=tolerance, 
                sources=sources
            )
            self.results['mem_tmc'] = np.concatenate([
                self.results['mem_tmc'], 
                np.reshape(marginals, (1,-1))
            ])
            self.results['idxs_tmc'].append(idxs)
        
    def tmc_iteration(self, tolerance, sources=None):
        """Runs one iteration of TMC-Shapley algorithm."""
        num_classes = len(set(self.y_test))
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}
        idxs = np.random.permutation(len(sources))
        marginal_contribs = np.zeros(len(self.X))
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        y_batch = np.zeros(0, int)
        truncation_counter = 0
        new_score = self.random_score
        for n, idx in enumerate(idxs):
            old_score = new_score
            X_batch = np.concatenate([X_batch, self.X[sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[sources[idx]]])
            if self.sample_weight is None:
                sample_weight_batch = None
            else:
                sample_weight_batch = np.concatenate([
                    sample_weight_batch, 
                    self.sample_weight[sources[idx]]
                ])
            if len(set(y_batch)) != num_classes and not self.is_regression:
                continue
            self.restart_model()
            #try:
            if True:
                if sample_weight_batch is None:
                    self.model.fit(X_batch, y_batch)
                else:
                    self.model.fit(
                        X_batch, 
                        y_batch,
                        sample_weight = sample_weight_batch
                    )
                new_score = self.value(self.model, metric=self.metric)    
            #except:
                #continue
            marginal_contribs[sources[idx]] = new_score - old_score
            marginal_contribs[sources[idx]] /= len(sources[idx])
            distance_to_full_score = np.abs(new_score - self.mean_score)
            if distance_to_full_score <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    print('Truncated at {}'.format(n))
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs
    
    def _calculate_loo_vals(self, sources=None, metric=None):
        """Calculated leave-one-out values for the given metric.
        
        Args:
            metric: If None, it will use the objects default metric.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
        
        Returns:
            Leave-one-out scores
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources==i)[0] for i in set(sources)}
        print('Starting LOO score calculations!')
        if metric is None:
            metric = self.metric 
        self.restart_model()
        if self.sample_weight is None:
            self.model.fit(self.X, self.y)
        else:
            self.model.fit(self.X, self.y,
                          sample_weight=self.sample_weight)
        baseline_value = self.value(self.model, metric=metric)
        vals_loo = np.zeros(len(self.X))
        for i in sources.keys():
            print(i)
            X_batch = np.delete(self.X, sources[i], axis=0)
            y_batch = np.delete(self.y, sources[i], axis=0)
            if self.sample_weight is not None:
                sw_batch = np.delete(self.sample_weight, sources[i], axis=0)
            if self.sample_weight is None:
                self.model.fit(X_batch, y_batch)
            else:
                self.model.fit(X_batch, y_batch, sample_weight=sw_batch)
                
            removed_value = self.value(self.model, metric=metric)
            vals_loo[sources[i]] = (baseline_value - removed_value)
            vals_loo[sources[i]] /= len(sources[i])
        return vals_loo
    
    def _concat(self, results, key, batch_result):
        
        if 'mem' in key or 'idxs' in key:
            if key in results.keys():
                if isinstance(results[key], list):
                    results[key].extend(batch_result)
                    return results[key]
                else:
                    return np.concatenate([results[key], batch_result])
            else:
                return batch_result.copy()
        else:
            if key in results.keys():
                return results[key] + batch_result
            else:
                return batch_result
    
    def _load_batch(self, batch_dir):
        
        try:
            batch = pkl.load(open(batch_dir, 'rb'))
            batch_sizes = [len(batch[key]) for key in batch if 'mem' in key]
            return batch, np.max(batch_sizes)
        except:
            return None, None
    
    def _filter_batch(self, batch, idxs=None):
        
        if idxs is None:
            return batch
        for key in batch.keys():
            if 'mem' in key or 'idxs' in key:
                if not isinstance(batch[key], list) and len(batch[key]):
                    batch[key] = batch[key][:, idxs]
        return batch
    
    def dist_stats(self, truncation, idxs=None):

        stats = {}
        if idxs is None:
            idxs = np.arange(len(self.X))
        stats['vals'] = np.zeros((len(idxs), truncation))
        stats['vars'] = np.zeros((len(idxs), truncation))
        stats['counts'] = np.zeros((len(idxs), truncation))
        batch_dirs = [os.path.join(self.directory, item) 
                   for item in os.listdir(self.directory)
                   if '_result.pkl' in item]
        for i, batch_dir in enumerate(np.sort(batch_dirs)):
            batch, batch_size = self._load_batch(batch_dir)
            if batch is None or batch_size == 0:
                continue
            if 'idxs_dist' not in batch.keys() or not len(batch['idxs_dist']):
                continue
            present = (batch['mem_dist'] != -1).astype(float)
            counts = np.array([len(i) for i in batch['idxs_dist']])
            for i, count in enumerate(counts):
                if count >= truncation:
                    continue
                present = (batch['mem_dist'][i, idxs] != -1).astype(float)
                stats['counts'][:, count] += present
                stats['vals'][:, count] += present * batch['mem_dist'][i, idxs]
                stats['vars'][:, count] += present * (batch['mem_dist'][i, idxs] ** 2)
        for i in range(len(stats['counts'])):
            nzs = np.where(stats['counts'][i] > 0)[0]
            stats['vals'][i, nzs] /= stats['counts'][i, nzs]
            stats['vars'][i, nzs] /= stats['counts'][i, nzs]
            stats['vars'][i, nzs] -= stats['vals'][i, nzs] ** 2
        return stats

    def load_results(self, max_samples=None, idxs=None, verbose=True):
        """Helper method for 'merge_results' method."""
        results = {}
        results_size = 0
        batch_dirs = [os.path.join(self.directory, item) 
                       for item in os.listdir(self.directory)
                       if '_result.pkl' in item]
        for i, batch_dir in enumerate(np.sort(batch_dirs)):
            batch, batch_size = self._load_batch(batch_dir)
            if verbose:
                print(batch_dir, batch_size)
            if batch is None or batch_size == 0:
                os.remove(batch_dir)
                continue
            if max_samples is not None:
                for key in batch:
                    if 'mem' in key or 'idxs' in key:
                        batch[key] = batch[key][:max_samples - results_size]
                results_size = min(results_size + batch_size, max_samples)
            batch = self._filter_batch(batch, idxs)
            for alg in set([key.split('_')[-1] for key in batch]):
                present = (batch['mem_' + alg] != -1).astype(float)
                del present
                if not len(batch['mem_' + alg]):
                    continue
                results['mem_' + alg] = self._concat(
                    results, 'mem_' + alg, batch['mem_' + alg])
                results['idxs_' + alg] = self._concat(
                    results, 'idxs_' + alg, batch['idxs_' + alg])
            if max_samples is not None and results_size >= max_samples:
                break
        self.results = results
    
    def merge_results(self, chunk_size=100):
        
        batch_dirs = np.sort([os.path.join(self.directory, item) 
                              for item in os.listdir(self.directory)
                              if '_result.pkl' in item])
        batch_sizes = [os.path.getsize(batch_dir) for batch_dir in batch_dirs]
        merged_size = 0
        merged_dirs = [[]]
        for batch_dir, batch_size in zip(batch_dirs, batch_sizes):
            merged_dirs[-1].append(batch_dir)
            merged_size += batch_size
            if merged_size > chunk_size * 1e6:
                merged_dirs.append([])
                merged_size = 0
        for i, batch_dirs in enumerate(merged_dirs):
            result_dic = '{}_result.pkl'.format(str(i).zfill(6))
            merged_dir = os.path.join(self.directory, result_dic)
            if len(batch_dirs) == 1 and batch_dirs[0] == merged_dir:
                print(merged_dir, 'exists')
                continue
            results = {}
            for batch_dir in batch_dirs:
                batch, batch_size = self._load_batch(batch_dir)
                if batch is None or batch_size == 0:
                    continue
                for alg in set([key.split('_')[-1] for key in batch]):
                    results['mem_' + alg] = self._concat(
                        results, 'mem_' + alg, batch['mem_' + alg])
                    results['idxs_' + alg] = self._concat(
                        results, 'idxs_' + alg, batch['idxs_' + alg])
                
            pkl.dump(results, open(merged_dir, 'wb'), protocol=4)
            for batch_dir in batch_dirs:
                if batch_dir != merged_dir:
                    os.remove(batch_dir)
            print(merged_dir)
            
    def portion_performance(
        self, idxs, plot_points, sources=None, X=None, y=None, sample_weight=None, verbose=False):
        """Given a set of indexes, starts removing points from 
        the first elemnt and evaluates the new model after
        removing each point."""
        if X is None:
            X = self.X
            y = self.y
            sample_weight = self.sample_weight
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources==i)[0] for i in set(sources)}
        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            if verbose:
                print('{} out of {}'.format(len(plot_points)-i+1, len(plot_points)))
            keep_idxs = np.concatenate([sources[idx] for idx 
                                        in idxs[plot_points[i-1]:]], -1)
            X_batch, y_batch = X[keep_idxs], y[keep_idxs]
            if sample_weight is not None:
                sample_weight_batch = self.sample_weight[keep_idxs]
            try:
                self.restart_model()
                if self.sample_weight is None:
                    self.model.fit(X_batch, y_batch)
                else:
                    self.model.fit(X_batch, y_batch,
                                  sample_weight=sample_weight_batch)
                scores.append(self.value(
                    self.model,
                    metric=self.metric,
                    X=self.X_heldout,
                    y=self.y_heldout
                ))
            except:
                scores.append(init_score)
        return np.array(scores)[::-1]
