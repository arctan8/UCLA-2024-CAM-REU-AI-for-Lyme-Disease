from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import plotly.express as px
import numpy as np
import inflect

import multiprocessing
from multiprocessing import Pool
from functools import partial

import os

import collections
from collections import namedtuple

tSNE_Result = namedtuple('tSNE_Result', ['perplexity', 'divergence', 'X_transf'])
N_COMP = 2
INIT = 'pca'
def find_perplexity(X, start=5, stop=105, step=5, multipro=True, n_components=N_COMP, init=INIT, **kwargs):
    '''
    Finds best perplexity which results in lowest KL divergence for t-SNE.

    Params:
    start (int, default=5): start index of perplexity
    stop (int, default=105): stop index of perplexity
    step (int, default=5): step of perplexity
    n_components (int, default=2): dimension of t-SNE
    init (string, default='pca'): initialization for t-SNE
    title (string, default=''): name of experiment
    multipro (bool, default=True): use of multiprocessing

    Returns:
    best_result (tSNE_Result): best_perplexity, lowest_divergence, X_transf
    '''
    
    title = kwargs.get('title','')
    perplexity = np.arange(start, stop, step)

    lowest_divergence = np.inf
    best_perplexity = 0
    divergence = [] 

    if multipro:
        num_cpus = os.cpu_count()

        comb = [(X, p) for p in perplexity]
        
        with Pool(num_cpus) as pool:

            partial_func = partial(compute_tsne, n_components=n_components, init=init)

            results = pool.starmap(partial_func, comb)

            best_result = min(results, key=lambda r: r.divergence)
            divergence = [r.divergence for r in results]
            
    else:
        for i in perplexity:
            # model = TSNE(n_components=2, init=init, perplexity=i)
            # reduced = model.fit_transform(X)
            # div = model.kl_divergence_
            # divergence.append(div)

            tsne_result = compute_tsne(X, p=i, n_components=n_components, init=init)
            div = tsne_result.divergence
            divergence.append(div)
            X_transf = tsne_result.X_transf
            
            if div < lowest_divergence:
                lowest_divergence = div
                best_perplexity = i
                
        best_result = tSNE_Result(perplexity=best_perplexity, divergence=lowest_divergence, X_transf=X_transf)

    fig = px.line(x=perplexity, y=divergence, markers=True)

    xtitle = title + " Perplexity"
    ytitle = title + " Divergence"
    
    fig.update_layout(xaxis_title=xtitle, yaxis_title=ytitle)
    fig.update_traces(line_color="red", line_width=1)
    fig.show()
    
    return best_result

def compute_tsne(X, p, n_components=N_COMP, init=INIT):
    '''
    Compute t-SNE on data X with perplexity p. Returns a tSNE_Result containing perplexity and divergence.

    Params:
    X (np.array): data
    p (int): perplexity
    n_components (int, default=2): dimensionality of reduced dataset
    init (string, default='pca'): default axes to initialize t-SNE
    
    Return:
    tSNE_Result: perplexity, divergence
    '''
    model = TSNE(n_components=n_components, perplexity=p, init=init)
    reduced = model.fit_transform(X)
    div = model.kl_divergence_

    return tSNE_Result(perplexity=p, divergence=div, X_transf=reduced)

def display_best_tsne(X_transf, Y, x_axis_index=0, y_axis_index=1, **kwargs):
    '''
    Displays t-SNE plot.
    
    Params:
    X (np.array): data array on which tsne has been applied, reduced dimension to >= 2.
    Y (np.array): label array. Can be one-hot-encoded or vectorized.
    x_axis_index (int, optional): index of first t-SNE component.
    y_axis_index (int, optional): index of second t-SNE component.

    Returns:
    None
    '''
    title = kwargs.get('title','')
    
    # convert Y to one-hot encoding
    if Y.ndim > 1:
        # Convert One-hot-encoding to vectorizing
        Y = np.argmax(Y, axis=1)

    assert X_transf.shape[0] >= 2, "Reduction to at least 2 dimensions via t-SNE!"
    
    fig = px.scatter(x=X_transf[:, x_axis_index],y=X_transf[:, y_axis_index], color=Y)

    # Write 1st, 2nd, etc
    p = inflect.engine()
    x_axis_comp = p.ordinal(x_axis_index + 1)
    y_axis_comp = p.ordinal(y_axis_index + 1)
    fig.update_layout(
        title=f"{title} t-SNE Visualization",
        xaxis_title=f"{x_axis_comp} component t-SNE",
        yaxis_title=f"{y_axis_comp} component t-SNE"
    )
    fig.show()

