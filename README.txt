1. Loading Data with LymeData:

To load the preprocessed data in '/home/reu23/jingyi/DATA.zip' and
filter the data, import the LymeData class:

import lymedata
from lymedata import *

Import constants:
import constants
from constants import *

Initialize the LymeData class as follows: 
a) Specify a certain group of patients (chronic, acute, neuro, musculo) with select_rows. All patients under consideration must be
specified, whether or not their group is used for classification. eg. specify only neuro for neuro vs non-neuro; neuro, musculo for neuro vs musculo.
b) Specify the symptoms, diagnostic circumstances, additional diagnostic circumstances, categorical circumstances with select_cols. Specific
circumstances may be specified using the individual_cols parameter.
c) Specify the labels by which classification will take place
(chronic, acute, neuro, non-neuro, musculo, non-musculo).
d) The drop_99 flag indicates whether to drop 99 values.

Any row with NaN values in the columns determing Neuro/Musculo status is dropped. 
For diagnostic circumstances, NaN values are filled with 0. 99 values may be dropped depending on flag drop_99.
For symptoms, NaN values are dropped by default. 
Patients answering less than 8 questions can be filtered and NaN filled with 0 depending on flag drop_skipped_8 (default True).

lyme_data = LymeData({ACUTE, MUSCULO},{DIAG_CIR},{MUSCULO, NON_MUSCULO}, drop_99=False)

Obtain the data and labels using get_data_and_labels. If drop_99 flag is False, the missing data matrix is also returned.

data_matrix, label_matrix, missing_data_matrix = lyme_data.get_data_and_labels()

If one wishes to specify only one label, such as in Jingyi's SSNMF, specify two labels in the initialization of the LymeData object and drop one after:

lyme_data = LymeData({CHRONIC, ACUTE, MUSCULO, NON_MUSCULO}, {SYMPTOMS},{MUSCULO, NON_MUSCULO}, drop_skipped_8=True)
lyme_data.drop_one_label(NON_MUSCULO)

2. SSNMF: Haddock SSNMF (NNLS)
a) Prof. Haddock's SSNMF algorithm is implemented by the Haddock_SSNMF class. Under the hood, it uses Pypi_SSNMF, which yields replicable results with same random seed.

import haddock_ssnmf
from haddock_ssnmf import Haddock_SSNMF

Initialize the Haddock_SSNMF class with data matrix X, label matrix Y, and optional missing data matrix W.
The class handles data splitting for all matrices.

ssnmf = Haddock_SSNMF(data_matrix, label_matrix, W=missing_data_matrix)

Haddock_SSNMF also implements Gridsearch for the number of topics (k),lambda, and random state:
best_accuracy, best_params = ssnmf.gridsearch(param_range={'k': range(2,7),'lambda': list(np.linspace(0,1,10)), 'random_state': range(0,10)})

If desired, the boxplots for accuracy distribution (get_accu_distr) and reconstruction error distribution (get_reconerr_distr) can also
be obtained:
best_accuracy, best_params, accu_distr, Xreconerr_distr, Yreconerr_distr, Xtest_reconerr_distr = ssnmf.gridsearch(param_range={'k': range(2,7),'lambda': list(np.linspace(0,1,10)), 'random_state': range(0,10)},
                                                          get_topic_accu_distr=True, get_reconerr_distr=True)
display_distr(accu_distr)
display_distr(Xreconerr_distr, ylabel='||X-AS|| reconerrs')
display_distr(Yreconerr_distr, ylabel='||Y-BS|| reconerrs')
display_distr(Xtest_reconerr_distr, ylabel='||X-AS|| reconerrs')

Haddock_SSNMF also computes accuracy on the test set:
test_accuracy = ssnmf.test(best_params)

b) Jingyi's SSNMF is implemented by Jingyi_SSNMF. Unlike other SSNMF, it uses only one column of labels, and it computes f1 score and accuracy. It does not do cross validation or gridsearch. It uses ssnmf.SSNMF from the pypi package, not the SSNMF class defined in ssnmf_abstract_classes.py.

import jingyi_ssnmf
from jingyi_ssnmf import Jingyi_SSNMF

ssnmf = Jingyi_SSNMF(data, label)

The find_lambda function searches for the best lambda value given a value of k.

best_lambda, best_accuracy, best_f1 = ssnmf.find_lambda(k=4)
test_accuracy, test_f1 = ssnmf.test()

Please note that ssnmf.SSNMF does not have instance variable k, but Jingyi_SSNMF does. Specify this k in keyword argument when creating heatmaps:

display_ssnmf(model=ssnmf.best_model, feature_name=SYMPTOMS, feature_labels=DIAG_CIR_COLS.keys(), class_labels=lyme_data.labels,k=ssnmf.k)

c) Needell SSNMF (Non-NNLS):
Needell SSNMF is a non-nnls version of ssnmf. It avoids some convergence issues with nnls-ssnmf.
The parameters return values for Needell SSNMF and Haddock SSNMF are the same.
Further investigation required to analyze the accuracy difference between this and Haddock ssnmf.

import needell_ssnmf
from needell_ssnmf import Needell_SSNMF

d) Fulldatasearch (Needell and Haddock SSNMF): CAUTION: BUGS IN NEEDELL SSNMF FULLDATASEARCH
Full data search conducts training on the full training set; no cross validation.
Note: The training accuracy of a fulldata search is the accuracy of a trained model on data that it has already seen.
Do not compare the training accuracy of a fulldatasearch and a cross-validated gridsearch (which calculates the accuracy on the
unseen 5th fold of the data). Compare the testing accuracy of fulldatasearch and a cross-validated training gridsearch accuracy.

Using Needell SSNMF: specify split_train_test as False to indicate that the whole data matrix is provided. Specify train and test
indices for Needell ssnmf to use as a mask to hide test data:

ssnmf = Needell_SSNMF(X,Y, split_train_test=False, train_indices=train_indices, test_indices=test_indices)

Using Haddock SSNMF: specify split_train_test as False to indicate that the whole data matrix is provided. No need to specify
train/test indices.

ssnmf = Haddock_SSNMF(X,Y, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, split_train_test=False)

Fulldata search returns 3 values: a dictionary of training results, a dictionary of test results, and a dictionary of recostruction
error distributions (optional flag get_reconerr_distr):

train_results, test_results, reconerr_results = ssnmf.fulldatasearch(param_range={'k': range(3,5),'lambda': np.linspace(0,1,10), 'random_state': range(0,10)}, get_topic_accu_distr=True, get_reconerr_distr=True)

best_train_accu = train_results['best_train_accu']
best_train_param = train_results['best_train_param']
train_accu_distr = train_results['train_accu_distr']

best_test_accu = test_results['best_test_accu']
best_test_param = test_results['best_test_param']
test_accu_distr = test_results['test_accu_distr']

print('Best train: ', best_train_accu)
print('Best test: ', best_test_accu)

display_distr(train_accu_distr, ylabel='Train Accuracy')
display_distr(test_accu_distr, ylabel='Test Accuracy')

Xreconerr_distr = reconerr_results['Xreconerr_distr']
Yreconerr_distr = reconerr_results['Yreconerr_distr']

display_distr(Xreconerr_distr, ylabel='||X-AS|| reconerrs')
display_distr(Yreconerr_distr, ylabel='||Y-BS|| reconerrs')

For futher examples: see alex/code_lib/experiment/synthetic_traintest.

3. Displaying Heatmaps
For any SSNMF, the matrices A, B, and K may be displayed as follows:

import display_ssnmf
from display_ssnmf import *

best_model = ssnmf.best_model

The function display_ssnmf takes in several parameters: the name of the feature, the labels, and the classification labels.

display_ssnmf(model=best_model, feature_name=DIAG_CIR, feature_labels=list(DIAG_CIR_COLS.keys()), class_labels=[MUSCULO, NON_MUSCULO])

4. Displaying Statistics
For any LymeData object, the relative frequencies for each feature may be computed and displayed as follows:

import display_stats
from display_stats import *

data = LymeData({CHRONIC, NEURO, NON_NEURO},{DIAG_CIR},{NEURO, NON_NEURO}, individual_cols={'tick born coinfection',
       'Babesia', 'Bartonella', 'Ehrlichia/ Anaplasma', 'Mycoplasma',
       'Rickettsia'}, defn=DEF_CNS1, drop_99=True)

The function display_stats takes in several parameters: the LymeData object, the xlabel, and optionally, the title.

display_stats(data, xlabel='Relative Frequency of Response "1"', ylabel='Variables')

5. Synthetic Data
Generate synthetic data with a specified sample size, test size, number of features, number of labels.
The relationship between labels and features is specified; it can be different for training labels and testing
labels, to simulate training set with low relation to testing set (test for overfitting).

gen = GenerateTrainTest(total_samples=1500, test_size=0.2, n_features=20, n_labels=2, shuffle=True)

Specify what percentage of the dataset is labeled as each; below is a 50/50 split between labels 0 and 1:
label_distr = {0: 0.5, 1:0.5}

Specify what labels rely on what features; below, label 0 consists of 30% each of features 1,2,3, label 1
consists of 30% each of features 3,4,5.

label_features = {0: {0: 0.3, 1: 0.3, 2: 0.3},
              1: {3: 0.3, 4: 0.3, 5: 0.3}}

Obtain separate training and testing matrices for Haddock SSNMF

X_train, X_test, Y_train, Y_test = gen.generateTrainTest(label_distr, train_label_features=label_features, test_label_features=label_features, train_random_size=0.3, test_random_size=0.3)

Obtain combined training and testing matrix, and train/test indices for Needell SSNMF

X, Y, train_indices, test_indices = gen.getTrainTestIndices()

Save dataset in npz file, with all matrices/indices saved:

np.savez('dataset1.npz', X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
         X=X, Y=Y, train_indices=train_indices)

