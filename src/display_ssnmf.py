import seaborn as sns
import matplotlib.pyplot as plt

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
    
    K = B @ A.T
    
    scale = 1
    
    figsize_A = (scale * A.shape[1], 0.75*scale * A.shape[0])
    figsize_B = (scale * B.shape[1], 0.75*scale * B.shape[0])
    figsize_K = (scale * K.T.shape[1], 0.75*scale * K.T.shape[0])

    display_heatmap(A, 'A', figsize_A, k, feature_name, feature_labels, defn, class_labels=class_labels)
    display_heatmap(B, 'B', figsize_B, k, 'Labels', class_labels, defn, class_labels=class_labels)
    display_heatmap(K.T, 'K', figsize_K, len(K), feature_name, feature_labels, defn, xlabel='Labels', **kwargs)
    
    
def display_heatmap(matrix, matrix_name, figsize, num_topics, feature_name, feature_labels, defn, **kwargs):
    # fig, ax = plt.subplots(figsize=figsize, dpi=100, constrained_layout=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    xlabel = kwargs.get('xlabel','Topics')

    xticklabels = kwargs.get('xticklabels', range(1,num_topics+1))
    sns.heatmap(matrix, xticklabels=xticklabels, yticklabels=feature_labels)

    if isinstance(xticklabels, list):
        plt.xticks(rotation=45)
        
    # if feature_name == 'Labels':
    # plt.yticks(rotation=90) # for some strange reason the labels on matrix B are vertical
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(feature_name)
    
    title = f"{defn}: Matrix {matrix_name}: {feature_name} x Topics"
    
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    groupcounts = kwargs.get('groupcounts', None)

    if groupcounts is not None:
        count_text = "\n".join([f"{k}: {v}" for k, v in groupcounts.items()])
        ax.text(1.05, 1, count_text, transform=ax.transAxes,
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

    