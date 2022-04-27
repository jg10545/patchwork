# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import panel as pn
import scipy.sparse
import sklearn.neighbors

from patchwork._sample import PROTECTED_COLUMN_NAMES
from patchwork._trainmanager import _hist_fig, _empty_fig

def _get_weighted_adjacency_matrix(features, n_neighbors=10, temp=0.01):
    """
    
    """
    neighbors = sklearn.neighbors.NearestNeighbors(metric="cosine").fit(features)
    distances, indices = neighbors.kneighbors(features, n_neighbors=n_neighbors+1,
                                          return_distance=True)
    # first column will be the query itself with distance 0. prune 
    # that out.
    distances = distances[:,1:]
    indices = indices[:,1:]
    # compute weights (equation 2 from paper)
    # note that I added a negative sign in the exponent so that
    # closer vectors are weighed higher
    weights = np.exp(-1*distances/temp)
    
    row_indices = np.stack([np.arange(indices.shape[0])]*indices.shape[1],1)
    A = scipy.sparse.csr_matrix((weights.ravel(), (row_indices.ravel(),
                                              indices.ravel())))
    
    D_inv_sqrt = scipy.sparse.diags(1/np.sqrt(weights.sum(1)))
    W_norm = D_inv_sqrt*A*D_inv_sqrt
    return W_norm

def _propagate_labels(df, W_norm, t=20):
    """
    
    """
    classes = [c for c in df.columns if c not in PROTECTED_COLUMN_NAMES]
    Y = {}
    training = (~df.exclude.values)&(~df.validation.values)
    for c in classes:
        Y[c] = 0.5*np.ones(len(df))
        subset = training&pd.notnull(df[c]).values
        hardlabels = df[c][subset].values
        Y[c][subset] = hardlabels
    
        for _ in range(t):
            Y[c] = W_norm.dot(Y[c])
            Y[c][subset] = hardlabels
          
    pred_df = pd.DataFrame(Y, index=df.index)
    return pred_df


class LabelPropagator():
    def __init__(self, pw):
        self.pw = pw
        
        # build widgets
        
        # for rebuilding adjacency matrix
        self._num_neighbors = pn.widgets.IntInput(name="Number of neighbors", value=10)
        self._temp = pn.widgets.FloatInput(name="Temperature", value=0.01)
        self._pred_batch_size = pn.widgets.LiteralInput(name='Prediction batch size', value=64, type=int)
        self._build_matrix_button = pn.widgets.Button(name="Build adjacency matrix")
        self._build_matrix_button.on_click(self._adjacency_matrix_callback)

        # and label propagation
        self._num_steps = pn.widgets.IntInput(name="Number of iterations", value=10)
        self._label_prop_button = pn.widgets.Button(name="Propagate Labels")
        self._label_prop_button.on_click(self._labelprop_callback)
        
        # objects to hold figures
        self._hist_fig = pn.pane.Matplotlib(_empty_fig(), width=500, height=300)
        self._hist_selector = pn.widgets.Select(name="Class", options=self.pw.classes)
        self._hist_watcher = self._hist_selector.param.watch(
                        self._hist_callback, ["value"])
        
    def panel(self):
        """
        Return code for label propagation panel
        """
        controls =  pn.Column(pn.pane.Markdown("### Weighted adjacency matrix"),
                              self._num_neighbors,
                              self._temp,
                              self._pred_batch_size,
                              self._build_matrix_button,
                              pn.layout.Divider(),
                              pn.pane.Markdown("### Label Propagation"),
                              self._num_steps,
                              self._label_prop_button)
        
        figures = pn.Column(pn.pane.Markdown("### Outputs By Class"),
                            self._hist_selector,
                            self._hist_fig)
        return pn.Row(controls, figures)
    
    def _hist_callback(self, *events):
        self._hist_fig.object = _hist_fig(self.pw.df, 
                                          self.pw.pred_df,
                                          self._hist_selector.value)
        
    def _adjacency_matrix_callback(self, *events):
        self.pw.build_nearest_neighbor_adjacency_matrix(
            self._pred_batch_size.value, 
            self._num_neighbors.value,
            self._temp)
    def _labelprop_callback(self, *events):
        self.pw.propagate_labels(self._num_steps.value)