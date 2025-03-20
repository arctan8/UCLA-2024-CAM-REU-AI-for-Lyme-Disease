# Jan 16 2024 backup
'''
Needell_SSNMF implements the algorithm outlined in https://arxiv.org/abs/2203.03551
Uses Pypi_SSNMF, extends SSNMF_Application.
Given data and labels, it uses the masking matrix L to withold a percentage of the labels for training and testing.
It does not split the data or label matrices in any way; all data and partial labels are used.
No NNLS.
It also computes the accuracy for a given set of parameters, to be used in gridsearch.
'''
import ssnmf_abstract_classes
import pypi_ssnmf
import numpy as np

from ssnmf_abstract_classes import SSNMF_Application
from pypi_ssnmf import Pypi_SSNMF

from sklearn.model_selection import train_test_split, KFold

class Needell_SSNMF(SSNMF_Application):
    SSNMF_TYPE = Pypi_SSNMF
    
    def __init__(self, X, Y, *args, **kwargs):
        '''
        Initializes Haddock_SSNMF.
        
        Params:
        X (np.array) data matrix, cases x features
        Y (np.array) label matrix, labels x features
        W (np.array, optional) missing data matrix, cases x features
        '''
        super().__init__(X,Y)
        
        X_rows = np.shape(X)[0] # rows = cases (eg patients)
        X_cols = np.shape(X)[1] # cols = features (eg symptoms)
        
        self.W = kwargs.get('W', np.ones((X_rows, X_cols)))
        self.train_indices, self.test_indices = self.train_test_split(test_size=0.2, random_state=42)

        self.L_test = self.get_L(self.train_indices) # mask for testing on SSNMF

    
    # Override SSNMF_Application.train_test_split
    def train_test_split(self, **kwargs):
        '''
        Despite the name, does not actually split the data and label matrices.
        Instead, returns a array of indices to be used in training and testing data.

        Params:
        test_size (optional): fraction of original data to be used for testing, default 0.2
        random_state (optional): random seed, default 42

        Returns: 
        train_indices (np.array): array of indices for training
        test_indices (np.array): array of indices for testing
        '''
        test_size = kwargs.get('test_size', 0.2)
        random_state = kwargs.get('random_state', 42)

        n = self.Y.shape[0] # number of rows
        row_indices = np.arange(n)
        
        train_indices, test_indices = train_test_split(row_indices, test_size=test_size, random_state=random_state)

        return train_indices, test_indices

    def get_accuracy(self, model, indices, **kwargs):
        '''
        Given a trained model and array of indices,
        calculate accuracy by comparing the predicted labels to the actual labels
        on those indices only.

        Params:
        model (Pypi_SSNMF): trained model
        indices (np.array): array of indices
        
        Returns:
        accuracy (float): accuracy score
        '''
        Y = model.Y.T
        y_len = len(indices)
        
        Y_hat = (model.B @ model.S).T
        Y_hat = np.round(Y_hat)
        
        assert Y_hat.shape == Y.shape

        correct_pred = 0 # True positive + True negative
        for i in indices:
            if (Y_hat[i] == Y[i]).all():
                correct_pred += 1
        accuracy = correct_pred / y_len # (TP + TN)/(TP+TN+FP+FN)
        
        return accuracy        

    # Override SSNMF_Application.cross_validate
    def cross_validate(self, param_values, kf=None, **kwargs):
        '''
        Compute the accuracy using cross-validation on Pypi_SSNMF
        with specified parameters.
        
        Parameters:
        kf (KFold): KFold object to split training data into folds
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF, default 1000
        '''
        N = kwargs.get('N', 1000)
        
        scores = []
        
        X_errs = []
        Y_errs = []
        # X_tst_errs = X_err since X is not split

        omit_indices_cv = self.test_indices # prevent testing indices from being selected in crossval
        cv_indices = self.train_indices # cval can only select from training indices
        
        Y_cv = self.Y[self.train_indices]

        for tr,tst in kf.split(Y_cv):
            train_index_cv = cv_indices[tr]
            test_index_cv = cv_indices[tst]

            L_train_cv = self.get_L(train_index_cv)
            
            model = Pypi_SSNMF(X=self.X.T, Y=self.Y.T, W=self.W.T, L=L_train_cv.T, 
                               k=param_values['k'], lam=param_values['lambda'], random_state=param_values['random_state'], modelNum=3)
            model.mult(numiters=N)

            accu = self.get_accuracy(model, test_index_cv)
            scores.append(accu)
            
            X_err = self.get_Xreconerr(model)
            X_errs.append(X_err)
    
            Y_err = self.get_Yreconerr(model)
            Y_errs.append(Y_err)

        avg_score = np.mean(scores)
        avg_X_reconerr = np.mean(X_errs)
        avg_Y_reconerr = np.mean(Y_errs)
        
        return avg_score, param_values,avg_X_reconerr, avg_Y_reconerr, avg_X_reconerr

    def test(self, *args, **kwargs):
        '''
        Using best parameter values calculated by gridsearch or specified values, train SSNMF on full training labels, set self.best_model, then tests it.
        
        Parameters:
        param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns:
        accuracy (float): calculated by sklearn.accuracy_score
        '''
        param_vals = kwargs.get('param_vals', self.best_param_vals)
        N = kwargs.get('N', 1000)
        
        if param_vals is None:
            raise Exception('Call Gridsearch before testing!')
            
        self.best_model = Pypi_SSNMF(X=self.X.T, Y=self.Y.T, W=self.W.T, L=self.L_test.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)
        
        self.best_model.mult(numiters=N)

        # Account for missing values in testing 
        return self.get_accuracy(self.best_model, self.test_indices)      

    def get_L(self, indices):
        ''' 
        Given a set of indices, create mask matrix L with same shape as Y.
        For each row (data point), all 1's indicate inclusion.
        All 0's indicate exclusion.

        Params:
        indices (np.array): array of indices marked for inclusion

        Returns:
        L (np.array): patients x label binary matrix, 1's or 0's in e/a row.
        '''
        Y_rows = np.shape(self.Y)[0] # rows = cases (patients)
        Y_cols = np.shape(self.Y)[1] # cols = number of classes (eg neuro vs non-neuro)
        
        L = np.zeros((Y_rows, Y_cols))
        L[indices,:] = 1

        return L

    def fulldata_validate(self, param_vals, **kwargs):
        '''
        Compute accuracy using Pypi_SSNMF on full testing dataset using testing mask, given a set of parameters.
        
        Parameters:
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF, default 1000
        
        Returns:
        score (float): accuracy score
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        Xreconerr (float): ||X-AS|| reconstruction error
        Yreconerr (float): ||X-AS|| reconstruction error
        '''
        N = kwargs.get('N', 1000)

        full_model = Pypi_SSNMF(X=self.X.T, Y=self.Y.T, W=self.W.T, L=self.L_test.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        full_model.mult(numiters=N)
        
        score =  self.get_accuracy(full_model, self.test_indices) 
        X_reconerr = self.get_Xreconerr(full_model)
        Y_reconerr = self.get_Yreconerr(full_model)

        return score, param_vals, X_reconerr, Y_reconerr
        
    def get_best_fulldata_model(self, **kwargs):
        '''
        Train Pypi_SSNMF on full dataset using the best parameters found for fulldata_validate.
        Parameters:
        fulldata_best_param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns: None
        '''
        N = kwargs.get('N', 1000) 
        param_vals = kwargs.get('fulldata_best_param_vals', self.fulldata_best_param_vals)
        
        self.best_fulldata_model = Pypi_SSNMF(X=self.X.T, Y=self.Y.T, W=self.W.T, L=self.L_test.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        self.best_fulldata_model.mult(numiters=N)
        return self.best_fulldata_model





        