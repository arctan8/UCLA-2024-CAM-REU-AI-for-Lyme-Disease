from collections import namedtuple

'''
namedtuples of results for SSNMF objects like haddock_ssnmf
'''
Experiment = namedtuple('Experiment', ['X_train', 'Y_train', 'X_test', 'Y_test', 'W_train', 'W_test'])
Param = namedtuple('Param', ['k','lambda_val','random_state'])

Crossvalidation_Result = namedtuple('Crossvalidation_Result', ['validation_score','param_vals','xreconerr','yreconerr','x_valreconerr'])
Gridsearch_Result = namedtuple('Gridsearch_Result', ['best_accuracy_overall','best_param_vals_overall','accu_distr', 'Xreconerr_distr', 'Yreconerr_distr', 'X_val_reconerr_distr', 'experiment'])
Train_Results = namedtuple('Train_Results',['train_score', 'param_vals','xtrain_reconerr','ytrain_reconerr','test_score', 'experiment'])
Test_Results = namedtuple('Test_Results', ['param_vals','experiment', 'test_score','X_test_error','model'])

Fulldatasearch_Results = namedtuple('Fulldatasearch_Results',['train_results','test_results','reconerr_results'])
