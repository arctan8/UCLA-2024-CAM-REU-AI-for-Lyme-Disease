import ssnmf_abstract_classes
from ssnmf_abstract_classes import SSNMF_Application
from functools import partial
import numpy as np

from ssnmf import SSNMF
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import nnls
from multiprocessing import Pool, cpu_count

class Jingyi_SSNMF(SSNMF_Application):
    def __init__(self, X, Y):
        if np.isnan(X).any() or np.isnan(Y).any():
            raise ValueError('No NaN values allowed in matrices.')
        if np.any(X == 99) or np.any(Y == 99):
            raise ValueError('No 99 values allowed in matrices.')
            
        super().__init__(X,Y)
        
        self.train_test_split()

    def fulldata_validate(self):
        raise Error("Not Implemented!")
        
    def train(self, lam, model, N):
        '''
        Helper function for find_lambda. Must be declared outside of find_lambda and return lamda value for parallel processing reasons.
        
        Parameters:
        lam (float): lambda value
        model (ssnmf.SSNMF): untrained model
        N (int): number of training iterations
        
        Returns:
        accuracy (float): accuracy score from self.get_accuracy
        f1 (float): f1 score from self.get_accuracy
        lam (float): parameter
        '''
        model.lam = lam
        model.mult(numiters=N)
        accuracy, f1 = self.get_accuracy(model, self.X_train.T, self.Y_train.T) 
        return accuracy, f1, lam
        
    def find_lambda(self, k, **kwargs):
        '''
        Finds lambda value that yields highest accuracy for fixed number of topics, k. 
        Uses entire training set, no cross-validation. Parallel Processing.
        
        Parameters:
        k (int, optional): number of topics, default 4
        N (int, optional): number of iterations, default 100
        lambda_values (np.array, optional): array of values to test, default from 0 to 1000, step 100
        
        Returns:
        best_lambda (float): lambda of highest accuracy
        best_accuracy (float): highest accuracy
        best_f1 (float): best f1 score
        '''
        self.k = kwargs.get('k',4)
        N = kwargs.get('N', 100)
        lambda_values = kwargs.get('lambda_values', np.linspace(0, 1000, num=100))
        lambda_values = list(lambda_values)
        
        model = SSNMF(X=self.X_train.T, k=self.k, modelNum=3, Y=self.Y_train.T, lam=1)
        
        num_cores = cpu_count()
        partial_func = partial(self.train, model=model, N=N)
        
        with Pool(num_cores) as pool: 
            results = pool.map(partial_func, lambda_values)
            best_result = max(results, key=lambda x: x[0])
            best_accuracy = best_result[0]
            best_f1 = best_result[1]
            best_lambda = best_result[2]
            
        self.best_param_vals = {'k': self.k, 'lambda': best_lambda}
        return best_lambda, best_accuracy, best_f1
    
    def get_accuracy(self, model, X_data, Y_labels):
        '''
        Find the accuracy and f1 score of the model. Obtains a predicted coefficient matrix from X_data using NNLS and compares the predicted labels to the actual labels.

        Parameters:
        model (ssnmf.SSNMF): trained model
        X_data (np.array): data array
        Y_labels (np.array): label array

        Returns:
        accuracy (float): accuracy score
        f1 (float): f1 score
        '''
        A = model.A
        B = model.B
        S = self.find_matrix_S(X_data, A)
        Y_hat = (B @ S)
        
        Y_hat_binary = np.where(Y_hat > 0.5, 1, 0) # convert Y_hat to binary
        Y_hat_binary = Y_hat_binary.reshape(-1)
        Y_labels = Y_labels.reshape(-1)
        
        f1 = f1_score(Y_labels, Y_hat_binary)
        accuracy = accuracy_score(Y_labels, Y_hat_binary)

        return accuracy, f1

    def test(self, *args, **kwargs):
        '''
        Train SSNMF on full training set, selts self.best_model, then tests it.

        Parameters:
        param_vals (dict, optional): k=k_val, lambda = lambda_val. Default is k and best lambda from find_lambda.

        Returns:
        accuracy (float): calculated by sklearn.accuracy_score
        f1 score (float): calculated by sklearn.f1_score
        '''
        param_vals = kwargs.get('param_vals', self.best_param_vals)
        if param_vals is None:
                raise Exception('Call find_lambda before testing!')

        self.best_model = SSNMF(X=self.X_train.T, k=self.best_param_vals['k'], modelNum=3, Y=self.Y_train.T, lam=self.best_param_vals['lambda'])
        
        N = kwargs.get('N', 1000)
        self.best_model.mult(numiters=N)
        
        return self.get_accuracy(self.best_model, self.X_test.T, self.Y_test.T)

    def find_matrix_S(self, X, A):
        num_samples, num_features = X.shape
        num_components = A.shape[1]
        S = np.zeros((num_components, num_features))
        for i in range(num_features):
            s_i, _ = nnls(A, X[:, i])
            S[:, i] = s_i
        return S                                   

    def cross_validate(self, param_values, kf, **kwargs):
        raise Exception("This SSNMF does not implement cross-validation!")

    def gridsearch(self, **kwargs):
        raise Exception("This SSNMF does not do gridsearch!")
        