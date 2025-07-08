import numpy as np

def four_to_double_label_multiclass(four_label):
    '''
    4-Label matrix simulates Neuro vs Musculo vs Both vs Neither.
    Double Label matrix simulates Double-Label Neuro vs Musculo.

    Given a 4-Label matrix with classes 0,1,2,3, return a double-Label matrix with 4 classes.
    Double-Label matrix:
    Class 2 is treated as Both classes 0 and 1, Class 3 is treated as Neither class 0 nor 1.

    Params:
    four_label (np.array): 4-Label matrix, n x 4

    Returns:
    double_label (np.array): double-label matrix, n x 2
    '''
    class_indices = np.argmax(four_label, axis=1)

    double_label = np.zeros((len(class_indices), 2), dtype=int)

    double_label[class_indices == 0] = [1,0]
    double_label[class_indices == 1] = [0,1]
    double_label[class_indices == 2] = [1,1]
    double_label[class_indices == 3] = [0,0]

    return double_label
