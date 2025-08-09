import seaborn as sns
import matplotlib.pyplot as plt

from tensor_utils import *
from sklearn.preprocessing import normalize

def display_ssnmf(model, feature_name, feature_labels, class_labels, defn, **kwargs):
    '''
    Display K = AB^T, A, and B heatmaps.
    
    Parameters:
    model (SSNMF): trained model
    feature_name (string): name of feature (eg Symptoms)
    feature_labels (list): list of feature names
    class_labels (list): list of class names
    k (int, optional): number of topics
    groupcounts (dict, optional): group count dictionary
    trial (int, optional): trial number
    
    figsize_A (tuple, optional): size of matrix A
    figsize_B (tuple, optional): size of matrix B
    figsize_K (tuple, optional): size of matrix K
    '''

    normal = kwargs.get('normal', False)
    A = model.A
    B = model.B
    
    if 'k' in kwargs.keys():
        k = kwargs.get('k')
    else:
        try:
            k = model.k
        except AttributeError:
            raise ValueError('This SSNMF version does not have an instance variable k, please specify k in kwargs.')
    
    K = B @ A.T

    # Convert K, A, B to numpy if necessary
    A = to_numpy(A)
    B = to_numpy(B)
    K = to_numpy(K)

    if normal == True:
        A = normalize(A.T, norm='l1').T
        print(np.sum(A, axis = 0))
        # B = normalize(B.T, norm='l1').T
        B = normalize(B, norm='l1')
        print(np.sum(B, axis = 0))
        K = normalize(K, norm='l1')
        print(np.sum(K.T, axis = 0))
    
    scale = 1
    
    figsize_A = (scale * A.shape[1], 0.75*scale * A.shape[0])
    figsize_B = (1.75* scale * B.shape[1], 1.75 * 0.75*scale * B.shape[0])
    figsize_K = (scale * K.T.shape[1], 0.75*scale * K.T.shape[0])

    display_heatmap(A, 'A', figsize_A, k, feature_name, feature_labels, defn, **kwargs)
    display_heatmap(B, 'B', figsize_B, k, 'Labels', class_labels, defn, **kwargs)
    display_heatmap(K.T, 'K', figsize_K, len(K), feature_name, feature_labels, defn, xlabel='Labels', **kwargs)
    
    
def display_heatmap(matrix, matrix_name, figsize, num_topics, feature_name, feature_labels, defn="", **kwargs):
    '''
    Displays and saves a heatmap. The y-axis name is given by feature_name, the y-axis ticks are given by feature_labels.
    The x-axis name is given by xlabel, the x-axis ticks are given by xticklabels.

    Params:
    matrix (np.array): matrix array
    matrix_name (str): name of matrix
    figsize (tuple): figure size, length x width
    num_topics (int): number of topics
    feature_name (str): name of features, eg. SYMPTOMS
    feature_labels (list): list of feature names, eg. SYMPTOMS_COLS
    defn (str, optional): definition name, eg. DEF_CNSA
    trial (int, optional): trial number
    '''
    
    trial = kwargs.get('trial', -1)
    normal = kwargs.get('normal', False)
    
    # fig, ax = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    xlabel = kwargs.get('xlabel','Topics')

    if matrix_name == 'A' or matrix_name == 'B':
        xticklabels = tuple(range(1,num_topics+1))
    else:
        xticklabels = kwargs.get('xticklabels', tuple(range(1,num_topics+1)))
                                 
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=feature_labels)

    if matrix_name == 'K' and len(xticklabels) > 1:
        plt.xticks(rotation=45)
        
    # if feature_name == 'Labels':
    # plt.yticks(rotation=90) # for some strange reason the labels on matrix B are vertical
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(feature_name)

    if trial == -1:
        title = f"{defn}: Matrix {matrix_name}: {feature_name} x Topics"
    else:
        title = f"{defn}, Trial {trial}: Matrix {matrix_name}: {feature_name} x Topics"

    # get normalized
    if normal:
        title += ", Normalized"
    
    ax.set_title(title)
    
    if len(feature_labels) > 1:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    groupcounts = kwargs.get('groupcounts', None)

    if groupcounts is not None:
        count_text = "\n".join([f"{k}: {v}" for k, v in groupcounts.items()])
        # ax.text(1.05, 1, count_text, transform=ax.transAxes,
        #         fontsize=10, verticalalignment='top',
        #         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
        ax.text(1.20, 1, count_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
    
    fig.savefig(f"{title}.png",bbox_inches='tight')
    plt.show()
    plt.close(fig)


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

    