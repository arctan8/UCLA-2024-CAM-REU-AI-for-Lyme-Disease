import numpy as np
import itertools
import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import KFold

# Abstract Classes (Interfaces)
'''
SSNMF class implements semi-supervised nonnegative matrix factorization.
It updates matrices by a multiplicative update algorithm.
'''
class SSNMF(ABC):
    # Parameters and their default ranges to Gridsearch
    PARAM_RANGE = dict() # param_name : list of param values
    
    def __init__(self, X, Y, k, *args, **kwargs):
        '''
        Instantiates abstract class SSNMF.
        
        Parameters:
        X (np.Array): data array, d features x n samples
        Y (np.Array): label array, c classes x n samples
        k (int): number of topics
        '''
        
        self.X = X
        self.Y = Y
        self.k = k # topics
        
        self.d = X.shape[0] # features
        self.n = X.shape[1] # samples
        self.c = Y.shape[0] # classes
    
    @abstractmethod
    def mult(self, numiters):
        '''
        Multiplicative update algorithm.
        Parameters:
        numiters (int): number of multiplicative update iterations
        Returns: None
        '''
        

'''
SSNMF_Application applies a given type of SSNMF in a certain way.
For this type of SSNMF, it is able to determine the accuracy given
parameter values. It is also able to conducts Gridsearch to determine the 
best parameter values.
'''
class SSNMF_Application(ABC):
    def __init__(self, X, Y):
        '''
        Instantiates abstract class SSNMF_Application.
        Parameters:
        X (np.array): full data matrix, features x samples
        Y (np.array): full label matrix, labels x samples
        '''
        self.X = X
        self.Y = Y
        self.best_param_vals = None
        self.best_model = None

        self.A = None
        self.B = None
        self.S = None        
    
    def train_test_split(self, **kwargs):
        '''
        Split data into training and testing sets.
        Parameters:
        test_size(float): fraction of data to be used for testing
        random_state(float): random seed
        '''
        test_size = kwargs.get('test_size', 0.2) # default is 0.8 train, 0.2 test 
        random_state = kwargs.get('random_state', 42)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test =  train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
    
    @abstractmethod
    def cross_validate(self, param_values, kf, **kwargs):
        '''
        Compute the accuracy using cross-validation on SSNMF
        with specified parameters.
        
        Parameters:
        kf (KFold): KFold object to split training data into folds
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF
        
        Returns:
        avg_score (float): accuracy score, averaged over all folds
        param_values (dict): dictionary, keys=param_name, vals=param_vals        
        avg_reconerr (float): reconstruction error, averaged over all folds
        '''
        
        
    def get_accuracy(self, model, X_data, Y_labels):
        '''
        Given a trained model and actual label matrix,
        calculate accuracy by comparing the predicted labels to the 
        actual labels.
        
        Parameters:
        model (SSNMF): trained model
        X_data (np.array): data to predict labels
        Y_labels (np.array): actual label matrix
        
        Return:
        accuracy (float): accuracy score
        '''
        raise NotImplementedError('Must implement accuracy function!')
    
    def get_Xreconerr(self, model):
        '''
        Given a trained model, compute reconstruction error ||X-AS||_F,
        using frobenius norm.

        Parameters:
        model (SSNMF): trained model

        Return:
        reconerr (float): reconstruction error
        '''
        
        return np.linalg.norm(np.multiply(model.W, model.X - model.A @ model.S), ord='fro')

    def get_Yreconerr(self, model):
        '''
        Given a trained model, compute reconstruction error ||X-AS||_F,
        using frobenius norm.

        Parameters:
        model (SSNMF): trained model

        Return:
        reconerr (float): reconstruction error
        '''
        
        return np.linalg.norm(np.multiply(model.L, model.Y - model.B @ model.S), ord='fro')
        
    def fulldatasearch(self,**kwargs):
        '''
        Fulldatasearch is a non-crossvalidated algorithm. For each combination of parameters,
        train on training set, test on testing set. Other than separation of data into training 
        and testing sets, no other separation of data (as with crossvalidation).

        Important distinction between fulldatasearch and gridsearch:
        
        Training accuracy, Fulldatasearch: train the SSNMF on the training data and seeing how well
        it predicts the training labels. 

        Training accuracy, Gridsearch: train the SSNMF on 4/5 of the training data and seeing how well
        it predicts the unseen 1/5 of the training labels. Cross-validation accuracy is like fulldatasearch
        testing accuracy.

        Testing accuracy, Fulldatasearch/Gridsearch: Using SSNMF trained on training data from above, see how
        well it predicts the testing labels. Same for both Fulldatasearch/Gridsearch.

        Default parameter range used from SSNMF.PARAM_RANGE. Specify parameter range
        using param_range. Uses parallel processing for higher speed.

        Parameters:
        param_range (list, optional): parameter range. Default: SSNMF_TYPE.PARAM_RANGE
        get_topic_accu_distr (bool, optional): if True, returns DataFrame of accuracies. Default False
        get_reconerr_distr (bool, optional): if True, returns DataFrame of reconstruction errors. Default False.

        Returns:
        best_accuracy (float): calculated by sklearn accuracy_score
        best_params (dict): dictionary, keys=param_name, vals=best_param_val

        train_results (dict): {'best_train_accu': (float), 'best_train_param': (DataFrame), 'train_accu_distr': (DataFrame)}

        test_results (dict): {'best_test_acc': (float), 'best_train_param': (DataFrame), 'test_accu_distr': (DataFrame)}

        reconerr_results (dict): {'Xreconerr_distr': (DataFrame), 'Yreconerr_distr': (DataFrame)}
        
        Notes: 
        train_accu_distr (DataFrame, optional): {'k': [accuracies]}
        test_accu_distr (DataFrame, optional): {'k': [accuracies]}
        Xreconerr_distr (DataFrame, optional): {'k': [reconerrs]}
        Yreconerr_distr (DataFrame, optional): {'k': [reconerrs]}
        '''
        
        param_range = kwargs.get('param_range', self.SSNMF_TYPE.PARAM_RANGE)
        param_names = list(param_range.keys())
        param_vals = list(param_range.values())

        get_topic_accu_distr = kwargs.get('get_topic_accu_distr', False)
        get_reconerr_distr = kwargs.get('get_reconerr_distr', False)
        
        comb = list(itertools.product(*param_vals)) # all possible combinations of params
        param_keys_and_comb = [dict(zip(param_names, s)) for s in comb]
        
        num_cores = cpu_count()
        
        best_accuracy_overall = 0
        best_param_vals_overall = dict()
        
        train_accu_distr = pd.DataFrame()
        test_accu_distr = pd.DataFrame()
        
        Xreconerr_distr = pd.DataFrame()
        Yreconerr_distr = pd.DataFrame()
        
        # Define partial function to call self.get_accuracy with the same kFold object kf
        partial_func = partial(self.fulldata_validate,**kwargs) 
        
        with Pool(num_cores) as pool:
            # pool.map returns a long list of tuples (accuracy_score, {param: vals}, X_reconerr, Y_reconerr, test_accuracy_score)
            results = pool.map(partial_func, param_keys_and_comb)    
            
            best_train_results = max(results, key=lambda x: x[0])
            best_train_overall = best_train_results[0]
            best_train_param_overall = best_train_results[1]


            # Buggy for some unknown reason
            # best_test_results = max(results, key=lambda x: x[4])
            test_data_extracted = [(test_acc, params) for (_, params, _,_, test_acc) in results]
            best_test_results = max(test_data_extracted, key=lambda x: x[0])
            best_test_overall = best_test_results[0]
            best_test_param_overall = best_test_results[1]

            print('best train results: ', best_train_results)
            print('best test results: ', best_test_results)
            
            if get_topic_accu_distr:
                for k in param_range['k']:
                    train_accu = [r[0] for r in results if r[1]['k'] == k]
                    train_accu_distr[k] = pd.Series(train_accu)

                    test_accu = [r[4] for r in results if r[1]['k'] == k]
                    test_accu_distr[k] = pd.Series(test_accu)

            if get_reconerr_distr:
                for k in param_range['k']:
                    Xreconerr = [r[2] for r in results if r[1]['k'] == k]
                    Xreconerr_distr[k] = pd.Series(Xreconerr)

                    Yreconerr = [r[3] for r in results if r[1]['k'] == k]
                    Yreconerr_distr[k] = pd.Series(Yreconerr)
            
        self.fulldata_best_train_param_vals = best_train_param_overall
        self.fulldata_best_train_model = self.get_best_fulldata_model() # model trained on self.X, self.Y, w/ best train params

        train_results = {'best_train_accu': best_train_overall, 'best_train_param': best_train_param_overall, 'train_accu_distr': train_accu_distr}

        test_results = {'best_test_accu': best_test_overall, 'best_test_param': best_test_param_overall, 'test_accu_distr': test_accu_distr}

        reconerr_results = {'Xreconerr_distr': Xreconerr_distr, 'Yreconerr_distr': Yreconerr_distr}
        
        if get_topic_accu_distr and get_reconerr_distr:
            return train_results, test_results, reconerr_results
        elif get_topic_accu_distr:
            return train_results, test_results
        elif reconerr_distr:
            return best_train_overall, best_train_param_overall, best_test_overall, best_test_param_overall, Xreconerr_distr, Yreconerr_distr
        else:
            return best_train_overall, best_train_param_overall, best_test_overall, best_test_param_overall

    def gridsearch(self, **kwargs):
        '''
        Conduct gridsearch with cross-validation over all parameters of the SSNMF. Default
        parameter range used from SSNMF.PARAM_RANGE. Specify parameter range
        using param_range. Uses parallel processing for higher speed.
        
        Parameters:
        random_state (int, optional): inital random seed for cross_validation split
        param_range (list, optional): parameter range. Default: SSNMF_TYPE.PARAM_RANGE
        get_topic_accu_distr (bool, optional): if True, returns DataFrame of accuracies. Default False
        get_reconerr_distr (bool, optional): if True, returns DataFrame of reconstruction errors. Default False.
        
        Returns:
        best_accuracy (float): calculated by sklearn accuracy_score
        best_params (dict): dictionary, keys=param_name, vals=best_param_val
        accu_distr (DataFrame, optional): {'k': [accuracies]}
        Xreconerr_distr (DataFrame, optional): {'k': [reconerrs]}
        Yreconerr_distr (DataFrame, optional): {'k': [reconerrs]}
        X_cvtst_reconerr_distr (DataFrame, optional): {'k': [reconerrs]}
        '''
        folds = kwargs.get('folds', 5)
        random_state = kwargs.get('random_state', 42)
        
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        
        param_range = kwargs.get('param_range', self.SSNMF_TYPE.PARAM_RANGE)
        param_names = list(param_range.keys())
        param_vals = list(param_range.values())

        get_topic_accu_distr = kwargs.get('get_topic_accu_distr', False)
        get_reconerr_distr = kwargs.get('get_reconerr_distr', False)
        
        comb = list(itertools.product(*param_vals)) # all possible combinations of params
        param_keys_and_comb = [dict(zip(param_names, s)) for s in comb]
        
        num_cores = cpu_count()
        
        best_accuracy_overall = 0
        best_param_vals_overall = dict()
        
        accu_distr = pd.DataFrame()
        Xreconerr_distr = pd.DataFrame()
        Yreconerr_distr = pd.DataFrame()
        X_cvtst_reconerr_distr = pd.DataFrame()
        
        # Define partial function to call self.get_accuracy with the same kFold object kf
        partial_func = partial(self.cross_validate, kf=kf, **kwargs)
        
        with Pool(num_cores) as pool:
            # pool.map returns a long list of tuples (accuracy_score, {param: vals})
            results = pool.map(partial_func, param_keys_and_comb)    
            best_result = max(results, key=lambda x: x[0])
            
            best_accuracy_overall = best_result[0]
            best_param_vals_overall = best_result[1]

            if get_topic_accu_distr:
                for k in param_range['k']:
                    accu = [r[0] for r in results if r[1]['k'] == k]
                    accu_distr[k] = pd.Series(accu)

            if get_reconerr_distr:
                for k in param_range['k']:
                    Xreconerr = [r[2] for r in results if r[1]['k'] == k]
                    Xreconerr_distr[k] = pd.Series(Xreconerr)

                    Yreconerr = [r[3] for r in results if r[1]['k'] == k]
                    Yreconerr_distr[k] = pd.Series(Yreconerr)

                    X_cvtst_reconerr = [r[4] for r in results if r[1]['k'] == k]
                    X_cvtst_reconerr_distr[k] = pd.Series(X_cvtst_reconerr)

        self.best_param_vals = best_param_vals_overall
        if get_topic_accu_distr and get_reconerr_distr:
            return best_accuracy_overall, best_param_vals_overall, accu_distr, Xreconerr_distr, Yreconerr_distr, X_cvtst_reconerr_distr
        elif get_topic_accu_distr:
            return best_accuracy_overall, best_param_vals_overall, accu_distr
        elif reconerr_distr:
            return best_accuracy_overall, best_param_vals_overall, Xreconerr_distr, Yreconerr_distr, X_cvtst_reconerr_distr
        else:
            return best_accuracy_overall, best_param_vals_overall
    
    @abstractmethod
    def fulldata_validate(self, **kwargs):
        '''
        Compute accuracy using Pypi_SSNMF on full dataset (self.X, self.Y), given a set of parameters.
        
        Parameters:
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF, default 1000
        
        Returns:
        score (float): accuracy score
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        Xreconerr (float): ||X-AS|| reconstruction error
        Yreconerr (float): ||X-AS|| reconstruction error
        '''
        
    @abstractmethod
    def test(self, *args, **kwargs):
        '''
        Test the SSNMF on the testing set.
        
        Parameters:
        param_vals (dict, optional): key=param_name, value=param_value
        
        Returns:
        accuracy (float): calculated by sklearn.accuracy_score
        '''
        
        