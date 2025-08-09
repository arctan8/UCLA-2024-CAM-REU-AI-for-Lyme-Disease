'''
Needell_SSNMF implements the algorithm outlined in https://arxiv.org/abs/2203.03551
Uses Pypi_SSNMF, extends SSNMF_Application.
Given data and labels, it uses the masking matrix L to withold a percentage of the labels for training and testing.
It does not split the data or label matrices in any way; all data and partial labels are used.
No NNLS.
It also computes the accuracy for a given set of parameters, to be used in gridsearch.
'''
import ssnmf_abstract_classes
import numpy as np

from tensor_utils import *
from ssnmf_utils import *
import torch

from ssnmf_abstract_classes import SSNMF_Application
from pypi_ssnmf import Pypi_SSNMF
from torch_ssnmf import Torch_SSNMF

from sklearn.model_selection import train_test_split, KFold

class Needell_SSNMF(SSNMF_Application):
    SSNMF_TYPE = Pypi_SSNMF
    
    def __init__(self, X, Y, split_train_test=True, train_indices=None, test_indices=None, torch=False, *args, **kwargs):
        '''
        Initializes Needell_SSNMF. If split_train_test is False, must provide train_indices, test_indices array.
        
        Params:
        X (np.array): data matrix, cases x features
        Y (np.array): label matrix, labels x features
        W (np.array, optional): missing data matrix, cases x features
        split_train_test (bool): boolean value to conduct random train-test-split. Default True.
        train_indices (np.array, optional): 1D array of training indices.
        test_indices (np.array, optional): 1D array of testing indices.
        '''
        super().__init__(X,Y, torch)
        
        X_rows = np.shape(X)[0] # rows = cases (eg patients)
        X_cols = np.shape(X)[1] # cols = features (eg symptoms)
        
        self.W = kwargs.get('W', np.ones((X_rows, X_cols)))
        if split_train_test:
            # self.train_indices, self.test_indices = self.train_test_split(test_size=0.2, random_state=42)
            self.train_indices, self.test_indices = self.train_test_split(test_size=0.2, **kwargs)
        else:
            assert train_indices is not None and train_indices.size > 0
            assert test_indices is not None and test_indices.size > 0
            self.train_indices = train_indices
            self.test_indices = test_indices            

        self.L_train = self.get_L(self.train_indices) # mask for testing on SSNMF

        if torch:
            self.ssnmf_app = Torch_SSNMF
        else:
            self.ssnmf_app = Pypi_SSNMF
        
        self.experiment = Needell_Experiment(X=self.X, Y=self.Y, L_train=self.L_train, train_indices=self.train_indices, test_indices=self.test_indices)
    
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
        model (SSNMF): trained model
        indices (np.array): array of indices
        
        Returns:
        accuracy (float): accuracy score
        X_tst_err (float): X reconstruction error
        '''
        X = to_numpy(model.X)
        Y = to_numpy(model.Y.T)
        A = to_numpy(model.A)
        B = to_numpy(model.B)
        S = to_numpy(model.S)
        W = to_numpy(model.W)

        X_tst_err = np.linalg.norm(np.multiply(W.T, (X - A @ S).T), ord='fro')
        
        y_len = len(indices)
        
        Y_hat = (B @ S).T
        
        assert Y_hat.shape == Y.shape
        
        # If Y has [1,1] or [0,0] then we are doing multiclass classification, eg NvM with implicit_both_neither
        # Else, binary classification, eg NvN_LABELS, MvM_LABELS, NvMvBvN_LABELS
        #     Instead of rounding, assign a 1 to whichever class has the highest value, and make the rest 0
        Y_hat_labels = None

        num_labels = Y.shape[1]
        
        # If Y_labels has [1,1] or [0,0] ie. NvM with implicit_both_neither
        if num_labels == 2 and (np.any(np.all(Y == 1, axis=1)) or np.any(np.all(Y == 0, axis=1))):
            Y_hat_labels = np.round(Y_hat)
        else: # binary classification
            result = np.zeros(Y_hat.shape)
            max_indices = np.argmax(Y_hat, axis=1)
            result[np.arange(len(result)), max_indices] = 1
            Y_hat_labels = result

        # correct_pred = np.sum(np.all(Y_labels == Y_hat_labels, axis=1))
        correct_pred = np.sum(np.all(Y[indices,:] == Y_hat_labels[indices,:], axis=1))

        # np.savetxt('Y_hat.txt', Y_hat[indices,:], fmt='%.2f')
        np.savetxt('Y_hat.txt', Y_hat, fmt='%.2f')
        np.savetxt('Y_hat_labels.txt', Y_hat_labels[indices,:], fmt='%.2f')
        np.savetxt('Y.txt', Y[indices,:], fmt='%.2f')

        accuracy = correct_pred / y_len
        return accuracy, X_tst_err

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
        X_tst_errs = []

        cv_indices = self.train_indices # cval can only select from training indices

        for tr, ts in kf.split(cv_indices):
            # tr, ts are lists of indices 0,1,... that correspond to indices of the cv_indices list.
            # they must be mapped back to values of the cv_indices list
            
            train_index_cv = cv_indices[tr]
            test_index_cv = cv_indices[ts] 

            # 1/5 of the training labels are masked (train_index_cv)
            # but all training data is availablie (cv_indices)
            L_train_cv = self.get_L(train_index_cv)            
            W_train_cv = self.get_W(cv_indices)
            
            # Note: technically speaking, none of these matrices need to be transposed
            # because SSNMF works the same way regardless. However since display_ssnmf re-transposes
            # matrix A and B, it is expedient to continue transposing for every matrix passed into ssnmf_app.
            # The reason for transposition is due to a legacy code misunderstanding which did not result result in
            # an actual bug.
            
            model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=W_train_cv.T, L=L_train_cv.T, 
                               k=param_values['k'], lam=param_values['lambda'], random_state=param_values['random_state'], modelNum=3)
            model.mult(numiters=N)

            accu, X_tst_err = self.get_accuracy(model, test_index_cv)
            scores.append(accu)
            X_tst_errs.append(X_tst_err)
            
            X_err = self.get_Xreconerr(model)
            X_errs.append(X_err)
    
            Y_err = self.get_Yreconerr(model)
            Y_errs.append(Y_err)

            del model
            torch.cuda.empty_cache()

        avg_score = np.mean(scores)
        avg_X_reconerr = np.mean(X_errs)
        avg_Y_reconerr = np.mean(Y_errs)
        avg_X_tst_err =  np.mean(X_tst_errs)
        
        return Crossvalidation_Result(avg_score, param_values,avg_X_reconerr, avg_Y_reconerr, avg_X_tst_err)

    def train(self, **kwargs):
        '''
        Conduct SSNMF on the full training data and labels to produce training accuracy. 
        Train SSNMF on the whole training dataset and evaluate it on that dataset.
        
        Parameters:
        param_vals (namedtuple): Param(k=, lambda_val=, random_state=)
        N (int, optional): number of iterations to train SSNMF, default 1000
        
        Returns:
        Train_Results (namedtuple), contains:
        
        train_score (float): training accuracy score
        param_vals (dict): dictionary, keys=param_name, vals=param_vals
        Xreconerr (float): ||X-AS|| reconstruction error
        Yreconerr (float): ||X-AS|| reconstruction error
        test_score (float): testing accuracy score
        '''
        N = kwargs.get('N', 1000)
        
        if self.best_param_vals is None:
            param_vals = kwargs.get('param_vals')
            assert param_vals is not None, "If no best parameters are found, please provide them for training!"
        else:
            param_vals = self.best_param_vals
        
        W_train = self.get_W(self.train_indices)

         # Train the model on all training data and all training labels
        full_model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=W_train.T, L=self.L_train.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)
    
        full_model.mult(numiters=N)
        # Here X_tst_err is the same as get_Xreconerr 

        # Get the accuracy of the model on the testing labels
        train_score, X_tst_err =  self.get_accuracy(full_model, self.test_indices) 
        X_reconerr = X_tst_err
        Y_reconerr = self.get_Yreconerr(full_model)
    
        return Train_Results(train_score, param_vals, X_reconerr, Y_reconerr, self.experiment)
        
    def test(self, *args, **kwargs):
        '''
        Using best parameter values calculated by gridsearch or specified values, train SSNMF on full training labels, set self.best_model, then tests it.
        
        Parameters:
        param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns:
        Test_Results (namedtuple), contains:
        
        accuracy (float): calculated by sklearn.accuracy_score
        x_tst_err (float): ||W_test \odot (X - AS) || where W_test is the mask matrix that selects only testing data
        '''
        param_vals = kwargs.get('param_vals', self.best_param_vals)
        N = kwargs.get('N', 1000)
        
        if param_vals is None:
            raise Exception('Call Gridsearch before testing!')

        # Train model on full data, withold testing labels
        self.best_model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=self.W.T, L=self.L_train.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)
        
        self.best_model.mult(numiters=N)

        # Account for missing values in testing 
        test_accuracy, _ =  self.get_accuracy(self.best_model, self.test_indices)

        W_test = self.get_W(self.test_indices) # Consider only testing indices
        X = to_numpy(self.best_model.X)
        S = to_numpy(self.best_model.S)
        A = to_numpy(self.best_model.A)

        x_tst_err = np.linalg.norm(np.multiply(W_test, (X - A @ S).T), ord='fro')

        return Test_Results(param_vals, self.experiment, test_accuracy, x_tst_err, self.best_model)

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

    def get_W(self, indices):
        ''' 
        Given a set of indices, create mask matrix W with same shape as X.
        For each row (data point), all 1's indicate inclusion.
        All 0's indicate exclusion.

        Params:
        indices (np.array): array of indices marked for inclusion

        Returns:
        X (np.array): patients x symptom binary matrix, 1's or 0's in e/a row.
        '''
        X_rows = np.shape(self.X)[0] # rows = cases (patients)
        X_cols = np.shape(self.X)[1] # cols = number of classes (eg neuro vs non-neuro)
        
        W = np.zeros((X_rows, X_cols))
        W[indices,:] = 1

        return W

    # Deprecated
    def fulldata_validate(self, param_vals, **kwargs):
        '''
        SSNMF on all (train+test) data and only training labels. The mask L only covers testing labels.
        
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

        full_model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=self.W.T, L=self.L_train.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        full_model.mult(numiters=N)

        train_score, X_tst_err = self.get_accuracy(full_model, self.train_indices)
        test_score,_ =  self.get_accuracy(full_model, self.test_indices) 
        
        X_reconerr = X_tst_err
        Y_reconerr = self.get_Yreconerr(full_model)

        return Train_Results(train_score, param_vals, X_reconerr, Y_reconerr, test_score, self.experiment)

    # Deprecated
    def get_best_fulldata_model(self, **kwargs):
        '''
        Train self.ssnmf_app with no mask (no data nor labels is hidden) using the parameters that produce best training accuracy,
        from fulldata_validate.
        
        Parameters:
        fulldata_best_param_vals (dict, optional): keys=param_names, values=param_vals
        
        Returns: None
        '''
        N = kwargs.get('N', 1000) 
        param_vals = kwargs.get('fulldata_best_param_vals', self.fulldata_best_train_param_vals)
        
        model = self.ssnmf_app(X=self.X.T, Y=self.Y.T, W=self.W.T, k=param_vals['k'], lam=param_vals['lambda'], random_state=param_vals['random_state'], modelNum=3)

        model.mult(numiters=N)
        return model





        