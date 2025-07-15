import matplotlib.pyplot as plt
import numpy as np

import constants
from constants import *

def display_stats(data, xlabel, ylabel, individual_cols=None, **kwargs):
    '''
    Plot histogram for statstics.

    Params:
    data (LymeData): LymeData object
    xlabel (str): label of x-axis
    ylabel (str): label of y-axis
    title (str, optional): title of plot
    gap (float, optional): gap between groups of bars for each variable. Default 0.25
    width (float, optional): width of bars. Default 0.25
    figsize (2-tuple, optional): size of figure
    '''

    gps = kwargs.get('gps', None)
    variables = [col for col in data.df.columns if col not in {NEURO, MUSCULO, BOTH, NEITHER, NON_NEURO, NON_MUSCULO}]

    # Labels used, eg {Neuro, Non-Neuro}
    select_cols = [col for col in data.df.columns if col in {NEURO, MUSCULO, BOTH, NEITHER, NON_NEURO, NON_MUSCULO}]

    if individual_cols is not None:
        variables = individual_cols

    variables.reverse() # Reverse variable list (for display purposes)
    # groups = {label: filtered dataframe with only rows with 1's in label}
    if gps == None:
        groups = {col: data.df[data.df[col] == 1] for col in select_cols}
    else:
        groups = {col: data.df[data.df[col] == 1] for col in gps}
    
    for col, group in groups.items():
        print(f'{col} group: {len(group)}')

    # counts = {label: list of counts for each variable}
    counts = {col: [group[var].sum()/len(group) for var in variables] for col, group in groups.items()}

    gap = kwargs.get('gap', 0.25)
    y = np.arange(len(variables)) * (1 + gap) # label locations
    width = kwargs.get('width', 0.5)

    # max default width: (6,5)
    figsize = kwargs.get('figsize', (max(len(variables)* width, 6), 5))
    fig, ax = plt.subplots(figsize=(figsize))

    k = 0
    total_cols = len(counts.keys())
    
    offset = (width / total_cols) * (total_cols - 1) / 2  # Centering offset
    for col, count in counts.items():
        ax.barh(y-offset + k*width/total_cols, count, width/total_cols, label=col, edgecolor='black')
        k += 1

    title = kwargs.get('title', xlabel + ' vs '+ ylabel)
    ax.set_title(title)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels(variables) # keep variable names horizontal
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()
        
    