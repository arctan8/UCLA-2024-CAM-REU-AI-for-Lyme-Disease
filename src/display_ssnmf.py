import seaborn as sns
import matplotlib.pyplot as plt

def display_ssnmf(model, feature_name, feature_labels, class_labels, **kwargs):
    '''
    Display K = AB^T, A, and B heatmaps.
    
    Parameters:
    model (SSNMF): trained model
    feature_name (string): name of feature (eg Symptoms)
    feature_labels (list): list of feature names
    class_labels (list): list of class names
    k (int, optional): number of topics
    
    figsize_A (tuple, optional): size of matrix A
    figsize_B (tuple, optional): size of matrix B
    figsize_K (tuple, optional): size of matrix K
    '''
    A = model.A
    B = model.B
    
    if 'k' in kwargs.keys():
        k = kwargs.get('k')
    else:
        try:
            k = model.k
        except AttributeError:
            raise ValueError('This SSNMF version does not have an instance variable k, please specify k in kwargs.')
        
    display_heatmap(A, 'A', (1.5, 5), k, feature_name, feature_labels)
    
    display_heatmap(B, 'B', (4, 1), k, 'Labels', class_labels)
    
    K = B @ A.T
    display_heatmap(K.T, 'K', (1, 8),len(K), feature_name, feature_labels, xlabel='Labels')
    
    
def display_heatmap(matrix, matrix_name, figsize, num_topics, feature_name, feature_labels, **kwargs):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, xticklabels=range(1,num_topics+1), yticklabels=feature_labels)
    
    xlabel = kwargs.get('xlabel','Topics')
    
    plt.xlabel(xlabel)
    plt.ylabel(feature_name)
    
    plt.title('Matrix '+ matrix_name + ': '+feature_name+' x Topics')
    plt.show()


def display_distr(distr, **kwargs):
    '''
    Plot boxplot distribution.
    
    Params:
    distr (DataFrame): {keys: [values]}
    xlabel (str, optional): label of x-axis. Default: k
    ylabel (str, optional): label of y-axis. Default: accuracy
    '''
    xlabel = kwargs.get('xlabel', 'k')
    ylabel = kwargs.get('ylabel','accuracy')

    sns.boxplot(data=distr)
    plt.title(f'{xlabel} vs {ylabel}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()

    