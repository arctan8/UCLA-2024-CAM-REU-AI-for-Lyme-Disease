Code Structure:
/code_lib
---/src
    Contains data filtering, SSNMF, heatmap display code, .py only
---/experiments
    All experiments, .ipynb only
---/tests
    All testing code for algorithms

File Structure:
SSNMF algorithm code
- pypi_ssnmf: implementation of SSNMF algorithm
-- PARAM_RANGE = {'k': range(2,7),'lambda': list(np.linspace(0,100,100)), 'random_state': range(0,10)}
- jingyi_ssnmf: untested, possibly needs debugging after port to new code library

SSNMF Accuracy algorithm code
- haddock_ssnmf: uses Pypi_SSNMF class, computes matrix S and accuracy, crossvalidation

Gridsearch/Testing code
- ssnmf_abstract_classes: 
-- SSNMF is parent class of all SSNMF variants (except jingyi_ssnmf)
-- SSNMF_Application is parent class of all accuracy algorithms
-----conducts train/test/split and gridsearch

Display code
- display_ssnmf: display the output of a ssnmf
- display_stats: display statistics for lymedata

Naming convention:
<Defn>_<Classes>_<Dataset>_<Technique>_<Params>

Defn: 
- CNS/PNS 1,2,3
- (D1) Dataset 1
- (D2) Dataset 2
- (D3) Dataset 3
Classes: 
- (NvN) Neuro vs Non-Neuro
- (NvM) Neuro vs Musculo
- (NvMvBvN) Neuro vs Musculo vs Both vs Neither
Dataset:
- (Cr) Circumstances
- (Sy) Symptoms
- (Sn) Synthetic
- (Ic) Isolated Circumstances: sick days and coinfections
- (Ac) Additional Circumstances
- (Cc) Categorical Circumstances

Technique:
- (H) Haddock
- (N) Needell
---- (s) Single-Label
---- (d) Double-Label
---- (4) 4-Label
---- (f) Fulldata
- (T) Tensorflow

Parameters: Only Specify if code has been executed.
- (k) topics. Specify range of k (eg k2-13, k6)
- (l) lambda. Specify range of lambda (eg l100 for lambda 0-100)
- (r) random states. Specify number of random states (eg r100 for random states 0-99)
- Default: k2-6, l1,r100

CSV File Naming:
<Name of ipynb file which created this CSV>_<Data Info>.csv

Data Info:
- (Accu) Accuracy
- (Xrec) X reconerrs
- (Yrec) Y reconerrs
- (XCVrec) X cross-validation reconerrs