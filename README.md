# UCLA 2024 CAM REU - AI in Lyme Disease Code  

This repository contains non-sensitive code developed for the UCLA 2024 CAM REU project on AI applications in Lyme disease research.  

## Contents  
- **Algorithms**: Implementations of Single-label, Double-label, and 4-Label SSNMF.  
- **SSNMF Variants**:  
  - **NNLS-based (Haddock) SSNMF**  
  - **Non-NNLS (Needell) SSNMF**  
- **Histogram Generation**: Code for visualizing results.  

## Authors
- PI: Prof. Deanna Needell, UCLA
- Alexandria Tan, UW '26
- Campbell Schelp, UCSD '26
## Notes  
- This repository does **not** include data or data processing code.  
- The focus is on algorithmic implementations and visualization tools.  

## Usage  
To use this project, clone the repository and install the dependencies (do not do this if using Lonepine):

```sh
git clone git@github.com:arctan8/UCLA-2024-CAM-REU-AI-for-Lyme-Disease 
```

Ensure dependencies are installed before running any scripts:  
```sh
pip install sklearn numpy seaborn pandas
```

## Code Example
Import the necessary packages:

```python
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import haddock_ssnmf
from haddock_ssnmf import Haddock_SSNMF

import display_ssnmf
from display_ssnmf import *

import numpy as np
```

Load the Iris Dataset. Ensure that the label matrix is one-hot-encoded, i.e it has one column for each of the
three classes, with only 1's and 0's in each column, where 1 indicates the class of the appropriate datapoint.

```python
iris = load_iris()

X = iris.data # Data matrix
Y = iris.target # Label matrix

Y = np.eye(3)[Y]
```

Initialize Haddock (NNLS) SSNMF with data and labels.

```python
haddock = Haddock_SSNMF(X, Y)
```

Conduct gridsearch over all possible combinations of `k` topics, tradeoff parameter $\lambda$, and random states.
Note that $\lambda$ is always between `[0,1]`.

```python
parameter_range = {'k': range(3,5),'lambda': np.linspace(0,1,10), 'random_state': range(0,10)}
best_accuracy_overall, best_param_vals_overall, accu_distr = haddock.gridsearch(param_range=parameter_range, get_topic_accu_distr=True)
```

Upon finding the best testing accuracy and obtaining the combination of parameters that yield this accuracy, test the SSNMF on the testing data.

```python
test_accuracy = haddock.test(best_param_vals_overall)

print(f'Best train accuracy overall {best_accuracy_overall}')
print(f"Best set of parameters: k: {best_param_vals_overall['k']}, lambda: {best_param_vals_overall['lambda']}, random_state: {best_param_vals_overall['random_state']}")
print(f'Best test accuacy: {test_accuracy}')
```

Display the accuracy distribution and the heatmap of the model with the highest training accuracy.

```python
display_distr(accu_distr, y_label='Train Accuracy')
display_ssnmf(model=haddock.best_model, feature_name='Iris Attributes', feature_labels=iris.feature_names, class_labels=iris.target_names)
```
