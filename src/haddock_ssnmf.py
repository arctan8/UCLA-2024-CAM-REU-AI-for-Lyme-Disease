# Formerly old_code_lib/pypi_haddock_ssnmf.py
'''
Haddock_SSNMF implements the algorithm outlined in https://doi.org/10.48550/arXiv.2010.07956
It uses Pypi_SSNMF and extends SSNMF_Application.
Given data, labels, and possibly a missing data binary matrix, it splits the dataset for training
and testing. It also computes the accuracy for a given set of parameters, to be used in gridsearch.
'''

import ssnmf_abstract_classes
import pypi_ssnmf
import numpy as np
import torch
import time

from tensor_utils import *
from ssnmf_utils import *

from ssnmf_abstract_classes import SSNMF_Application
from pypi_ssnmf import Pypi_SSNMF
from torch_ssnmf import Torch_SSNMF

from sklearn.model_selection import train_test_split
# from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from sklearn.metrics import accuracy_score


class Haddock_SSNMF(SSNMF_Application):
    SSNMF_TYPE = Pypi_SSNMF
    
    def __init__(self, X, Y, X_train=None, X_test=None, Y_train=None, Y_test=None, W_train=None, W_test=None, split_train_test=True, torch=False, *args, **kwargs):
        '''
        Initializes Haddock_SSNMF.
        
        Params:
        X (np.array) data matrix, cases x features
        Y (np.array) label matrix, labels x features
        W (np.array, optional) missing data matrix, cases x features
        '''
        super().__init__(X,Y, torch)
        
        X_rows = np.shape(X)[0] # rows = cases (eg patients)
        X_cols = np.shape(X)[1] # cols = features (eg symptoms)
        self.W = kwargs.get('W', np.ones((X_rows, X_cols)))
            
        if split_train_test:
            self.train_test_split(**kwargs)
        else:
            assert X_train is not None
            assert X_test is not None
            assert Y_train is not None
            assert Y_test is not None
            self.X_train = X_train
            self.X_test = X_test
            self.Y_train = Y_train
            self.Y_test = Y_test
            if W_train is None:
                self.W_train = np.ones((np.shape(X_train)[0], np.shape(X_train)[1]))
            else:
                self.W_train = W_train
                
            if W_test is None:
                self.W_test = np.ones((np.shape(X_test)[0], np.shape(X_test)[1]))
            else:
                self.W_test = W_test
        
        if torch:
            self.ssnmf_app = Torch_SSNMF
        else:
            self.ssnmf_app = Pypi_SSNMF

        self.experiment = Experiment(self.X_train, self.Y_train, self.X_test, self.Y_test, self.W_train, self.W_test)
        
    def find_matrix_S(self, X, A, **kwargs):
        '''
        Compute coefficient matrix S from data matrix X and basis matrix A using 
        Nonnegative Least Squares: X approx. AS.
        
        Parameters:
        X (np.array): data matrix, cases x features
        A (np.array): basis matrix, cases x topics
        W (np.array, optional): weight matrix, cases x features
        
        Returns:
        S (np.array or torch.Tensor): coefficient matrix, topics x features
        tol (float, optional): NNLS tolerance, default 1e-10
        '''
        # Convert to numpy array if X, A are tensors
        X = to_numpy(X)
        A = to_numpy(A)
        
        if X.shape[0] != A.shape[0]:
            raise Exception('Shape mismatch, X: ',X.shape,' A: ',A.shape)
        
        W = kwargs.get('W', np.ones(X.shape))
        W = to_numpy(W) if W is not None else None
        tol = kwargs.get('tol',1e-10)
        
        num_samples, num_features = X.shape
        num_components = A.shape[1]
        
        S = np.zeros((num_components, num_features))
        ### Decrease precision to avoid NNLS Non-convergence? Or reduce # of iterations
        
        for i in range(num_features):
            # s_i = None
            # if W is None:
            #     nnls_result = lsq_linear(A, X[:,i], bounds=(0,np.inf), tol=tol)
            #     s_i = nnls_result.x

            # else:
            #     W_i = W[:,i]
            #     W_i_matrix = np.diag(W_i)
            #     X_i = X[:,i]
                
            #     nnls_result = lsq_linear(W_i_matrix@A, W_i_matrix@X_i, bounds=(0, np.inf), tol=tol)
            #     s_i = nnls_result.x
            # S[:, i] = s_i
            W_i = None
            A_check = None
            X_check = None
            
            if W is None:
                A_check = A
                X_check = X[:, i]
            else:
                W_i = W[:, i]
                W_i_matrix = np.diag(W_i)
                A_check = W_i_matrix @ A
                X_check = W_i_matrix @ X[:, i]
            
            if np.isnan(A_check).any() or np.isinf(A_check).any():
                print(f"NaN or Inf detected in weighted A matrix at feature {i}")
            if np.isnan(X_check).any() or np.isinf(X_check).any():
                print(f"NaN or Inf detected in weighted X vector at feature {i}")
            if np.isnan(W_i).any() or np.isinf(W_i).any():
                print(f"NaN or Inf detected in weight vector at feature {i}")

            # clip very small weights to avoid zeros:
            if W is not None:
                W_i = np.clip(W_i, 1e-10, None)  # avoid zeros
                W_i_matrix = np.diag(W_i)
                A_check = W_i_matrix @ A
                X_check = W_i_matrix @ X[:, i]
                
            nnls_result = lsq_linear(A_check, X_check, bounds=(0, np.inf), method='bvls', tol=tol)
            s_i = nnls_result.x
            S[:, i] = s_i

        if self.torch:
            return torch.from_numpy(S)
        else:
            return S
    
    def get_accuracy(self, model, X_data, Y_labels, **kwargs):
        '''
        Given a trained model and actual label matrix,
        calculate accuracy by comparing the predicted labels to the 
        actual labels.
        
        Parameters:
        model (self.ssnmf_app): trained model
        X_data (np.array): data to predict labels
        Y_labels (np.array): actual label matrix (rows=patients, col=label)
        S (np.array, optional): coefficient matrix, specified for fulldata_validation or calculated via NNLS if unspecified.
        A (np.array, optional): matrix A
        W (np.array, optional): weight matrix W
        
        Returns:
        accuracy (float): accuracy score
        X_tst_err (float): ||X - AS|| for S generated by NNLS
        '''
        A = kwargs.get('A', model.A)
        A = to_numpy(A)
        
        B = model.B
        B = to_numpy(B)
        
        # S = model.S
        W = kwargs.get('W', None)
        if W is not None:
            W = to_numpy(W)
        
        # S = kwargs.get('S', self.find_matrix_S(X_data, A, W=W))
        start = time.time()
        S = kwargs.get('S', self.find_matrix_S(X_data, A, W=W))
        end = time.time()
        print(f'FIND MATRIX S: {end-start:.6f} seconds')
        
        if S is not None:
            S = to_numpy(S)
        
        X_tst_err = np.linalg.norm(X_data - A@S, ord='fro')
        
        Y_hat = B @ S
        
        # Assume that labels are mutually exclusive
        # Y_hat_labels = (np.argmax(Y_hat, axis=0) == 0).astype(int)
        # Y_labels = Y_labels[0] 
            
        # accuracy = accuracy_score(Y_labels, Y_hat_labels)
        if Y_labels.ndim == 1:
            Y_labels = np.column_stack((Y_labels, 1-Y_labels)) # Make Y_labels 2D
        
        Y_labels = Y_labels.T
        Y_hat = Y_hat.T        
        y_len = len(Y_hat)
        
        assert Y_hat.shape == Y_labels.shape
        
        # If Y_labels has [1,1] or [0,0] then we are doing multiclass classification, eg NvM with implicit_both_neither
        # Else, binary classification, eg NvN_LABELS, MvM_LABELS, NvMvBvN_LABELS
        #     Instead of rounding, assign a 1 to whichever class has the highest value, and make the rest 0\
        Y_hat_labels = None

        num_labels = Y_labels.shape[1]
        
        # if num_labels == 2 and (np.all(Y_labels == [1,1], axis=1).any() or np.all(Y_labels == [0,0], axis=1).any()):
        if num_labels == 2 and (np.any(np.all(Y_labels == 1, axis=1)) or np.any(np.all(Y_labels == 0, axis=1))):
            Y_hat_labels = np.round(Y_hat)
        else:
            result = np.zeros(Y_hat.shape)
            max_indices = np.argmax(Y_hat, axis=1)
            result[np.arange(y_len), max_indices] = 1
            Y_hat_labels = result
        
        # print(f'Y_hat_labels.shape: {Y_hat_labels.shape}')
        
        correct_pred = 0 # True positive + True negative
        
        # for true_label, pred_label in zip(Y_labels, Y_hat_labels):
        #     # print(f'true_label: {true_label}, pred_label: {pred_label}')
        #     if (true_label == pred_label).all():
        #         correct_pred += 1
        correct_pred = np.sum(np.all(Y_labels == Y_hat_labels, axis=1))
        
        accuracy = correct_pred / y_len # (TP + TN)/(TP+TN+FP+FN)
        return accuracy, X_tst_err
    
    def cross_validate(self, param_values, kf, **kwargs):
        '''
        Compute the accuracy using cross-validation on self.ssnmf_app
        with specified parameters.
        
        Parameters:
        kf (KFold): KFold object to split training data into folds
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF, default 1000
        
        Returns:
        avg_score (float): accuracy score, averaged over all folds
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        avg_Xreconerr (float): ||X-AS|| reconstruction error, averaged over all folds
        avg_Yreconerr (float): ||X-AS|| reconstruction error, averaged over all folds
        avg_X_tst_err (float): ||X_cv_tst -A S_nnls|| reconstruction error, averaged over all folds
        '''
        
        N = kwargs.get('N', 1000)
        
        scores = []
        X_errs = []
        Y_errs = []
        X_tst_errs = []
        
        for train_index, val_index in kf.split(self.X_train):
            X_train_cv, X_val_cv = self.X_train[train_index, :].T, self.X_train[val_index, :].T
            Y_train_cv, Y_val_cv = self.Y_train[train_index, :].T, self.Y_train[val_index, :].T
            W_train_cv, W_val_cv = self.W_train[train_index, :].T, self.W_train[val_index, :].T

            # X_train_cv: features x patients
            # Y_train_cv: features x labels
                
            model = self.ssnmf_app(X=X_train_cv, Y=Y_train_cv, W=W_train_cv, k=param_values['k'], lam=param_values['lambda'], random_state=param_values['random_state'], modelNum=3)
            # X_train_cv ~= AS, Y_train_cv ~= BS
            # model.A: features x topic
            # model.S: topic x patients
            # model.B: label x topic
            
            model.mult(numiters=N)

            start = time.time()
            
            
            accuracy, X_tst_err = self.get_accuracy(model,X_val_cv,Y_val_cv, W=W_train_cv) 

            end = time.time()
            print(f'GET ACCURACY: {end-start:.6f} seconds')
            
            scores.append(accuracy)
            X_tst_errs.append(X_tst_err)

            X_err = self.get_Xreconerr(model)
            X_errs.append(X_err)

            Y_err = self.get_Yreconerr(model)
            Y_errs.append(Y_err)

        avg_score = np.mean(scores) # self.get_accuracy returns floats
        avg_X_reconerr = np.mean(X_errs)
        avg_Y_reconerr = np.mean(Y_errs)
        avg_X_tst_err =  np.mean(X_tst_errs)
        
        # print(f'param_vals={param_values}, reconerr={avg_reconerr}')

        return Crossvalidation_Result(avg_score, param_values, avg_X_reconerr, avg_Y_reconerr, avg_X_tst_err)
        
    def test(self, *args, **kwargs):
        '''
        Using best parameter values calculated by gridsearch or specified values, train SSNMF on full training set, set self.best_model, then tests it.
        
        Parameters:
        param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns:
        accuracy (float): calculated by sklearn.accuracy_score
        X_tst_err (float): ||X_tst - A S_nnls ||
        '''
        param_vals = kwargs.get('param_vals', self.best_param_vals)
        N = kwargs.get('N', 1000)
        
        if param_vals is None:
            raise Exception('Call Gridsearch before testing!')
            
        self.best_model = self.ssnmf_app(X=self.X_train.T, Y=self.Y_train.T, W=self.W_train.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)
        
        self.best_model.mult(numiters=N)

        # Account for missing values in testing

        test_accuracy, x_tst_err = self.get_accuracy(self.best_model, self.X_test.T, self.Y_test.T, W=self.W_test.T)

        return Test_Results(param_vals, self.experiment, test_accuracy, x_tst_err, self.best_model)

    # Deprecated
    def fulldata_validate(self, param_vals, **kwargs):
        '''
        SSNMF on X_train, Y_train to produce Fulldata training accuracy. 
        Use factors A and B from training, conduct SSNMF on X_test and Y_test to produce testing accuracy.
        
        Parameters:
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        N (int, optional): number of iterations to train SSNMF, default 1000
        
        Returns:
        train_score (float): training accuracy score
        param_values (dict): dictionary, keys=param_name, vals=param_vals
        Xreconerr (float): ||X-AS|| reconstruction error
        Yreconerr (float): ||X-AS|| reconstruction error
        test_score (float): testing accuracy score
        '''
        N = kwargs.get('N', 1000)
        full_model = self.ssnmf_app(X=self.X_train.T, Y=self.Y_train.T, W=self.W_train.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        full_model.mult(numiters=N)
        # Here X_tst_err is the same as get_Xreconerr 
        train_score, X_tst_err =  self.get_accuracy(full_model, self.X_train.T, self.Y_train.T, S=full_model.S, W=self.W_train.T) 
        X_reconerr = X_tst_err
        Y_reconerr = self.get_Yreconerr(full_model)

        test_score, _ = self.get_accuracy(full_model, self.X_test.T, self.Y_test.T, W=self.W_test.T)

        # return train_score, param_vals, X_reconerr, Y_reconerr, test_score 
        return Train_Results(train_score, param_vals, X_reconerr, Y_reconerr, test_score, self.experiment)
        
    # Deprecated
    def get_best_fulldata_model(self, **kwargs):
        '''
        Train self.ssnmf_app on self.X, self.Y dataset (no data is hidden) using the parameters found for fulldata_validate.
        These parameters produce the highest training accuracy (not necessarily highest testing accuracy).
        Specify alternative parameters in optional argument.
        
        Parameters:
        fulldata_best_param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns: None
        '''
        N = kwargs.get('N', 1000) 
        param_vals = kwargs.get('fulldata_best_param_vals', self.fulldata_best_train_param_vals)
        
        model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=self.W.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        model.mult(numiters=N)
        return model
        
    def train_test_split(self, **kwargs):
        '''
        Split data into training and testing sets.
        
        Params:
        test_size (optional): fraction of original data to be used for testing, default 0.2
        random_state (optional): random seed, default 42
        
        '''
        test_size = kwargs.get('test_size', 0.2)
        random_state = kwargs.get('random_state', 42)

        self.X_train, self.X_test, self.Y_train, self.Y_test, self.W_train, self.W_test = train_test_split(self.X, self.Y, self.W, test_size=test_size, random_state=random_state)
