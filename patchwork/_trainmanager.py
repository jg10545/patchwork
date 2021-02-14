"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import tensorflow as tf
from sklearn.metrics import roc_auc_score




def _auc(pos, neg, rnd=3):
    """
    macro for computing ROC AUC score for display from arrays
    of scores for positive and negative cases
    """
    if (len(pos) > 0)&(len(neg) > 0):
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        y_pred = np.concatenate([pos, neg])
        return round(roc_auc_score(y_true, y_pred), rnd)
    else:
        return 0


def _empty_fig():
    """
    make an empty matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.plot([])
    ax.axis("off")
    plt.close(fig)
    return fig

def _loss_fig(l):
    """
    Generate matplotlib figure for a line plot of the 
    training loss
    """
    fig, ax = plt.subplots()
    ax.plot(l, "o-")
    ax.set_xlabel("step", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)
    ax.grid(True)
    plt.close(fig)
    return fig



def _hist_fig(df, pred, c):
    """
    
    """
    bins = np.linspace(0, 1, 15)
    unlabeled = pred[c][pd.isnull(df[c])].values
    
    fig, (ax1, ax2) = plt.subplots(2,1)
    
    # top plot: training data
    pos_labeled = pred[c][(df[c] == 1)&(df["validation"] == False)].values
    neg_labeled = pred[c][(df[c] == 0)&(df["validation"] == False)].values
    train_auc = _auc(pos_labeled, neg_labeled)
    if len(pos_labeled) > 0: 
        ax1.hist(pos_labeled, bins=bins, alpha=0.5, 
                    label="labeled positive (train)", density=True)
    if len(neg_labeled) > 0:
        ax1.hist(neg_labeled, bins=bins, alpha=0.5, 
                    label="labeled negative (train)", density=True)
    if len(unlabeled) > 0:
        ax1.hist(unlabeled, bins=bins, alpha=1., label="unlabeled", 
                 density=True, histtype="step", lw=2)
      
    # bottom plot: validation data
    pos_labeled = pred[c][(df[c] == 1)&(df["validation"] == True)].values
    neg_labeled = pred[c][(df[c] == 0)&(df["validation"] == True)].values
    test_auc = _auc(pos_labeled, neg_labeled)
    if len(pos_labeled) > 0:           
        ax2.hist(pos_labeled, bins=bins, alpha=0.5, 
                    label="labeled positive (val)", density=True)
    if len(neg_labeled) > 0:
        ax2.hist(neg_labeled, bins=bins, alpha=0.5, 
                    label="labeled negative (val)", density=True)
    if len(unlabeled) > 0:
        ax2.hist(unlabeled, bins=bins, alpha=1., label="unlabeled", 
                 density=True, histtype="step", lw=2)
    
    for a in [ax1, ax2]:
        a.legend(loc="upper left")
        a.set_xlabel("assessed probability", fontsize=14)
        a.set_ylabel("frequency", fontsize=14)
    title = "model outputs for '%s'\nAUC train %s, test AUC %s"%(c, train_auc, test_auc)
    ax1.set_title(title, fontsize=14)
    plt.close(fig)
    return fig




class TrainManager():
    """
    GUI for training the model
    """
    
    def __init__(self, pw):
        """
        :pw: patchwork object
        """
        self.pw = pw
        self._header = pn.pane.Markdown("#### Train model on current set of labeled patches")
        self._learn_rate =  pn.widgets.LiteralInput(name="Learning rate", value=1e-3, type=float)                               
        self._batch_size = pn.widgets.LiteralInput(name='Batch size', value=16, type=int)
        self._samples_per_epoch = pn.widgets.LiteralInput(name='Samples per epoch', value=1000, type=int)
        self._epochs = pn.widgets.LiteralInput(name='Epochs', value=10, type=int)
        
        self._eval_after_training = pn.widgets.Checkbox(name="Update predictions after training?", value=True)
        self._compute_badge_gradients = pn.widgets.Checkbox(name="Update BADGE gradients?", value=False)
        self._train_button = pn.widgets.Button(name="Make it so")
        self._train_button.on_click(self._train_callback)
        
        self._footer = pn.pane.Markdown("")
        
        # objects to hold figures
        self._loss_fig = pn.pane.Matplotlib(_empty_fig())
        self._hist_fig = pn.pane.Matplotlib(_empty_fig())
        self._hist_selector = pn.widgets.Select(name="Class", options=self.pw.classes)
        self._hist_watcher = self._hist_selector.param.watch(
                        self._hist_callback, ["value"])
        
    def panel(self):
        """
        Return code for a training panel
        """
        controls =  pn.Column(self._header,
                        self._learn_rate,
                         self._batch_size, 
                         self._samples_per_epoch,
                         self._epochs,
                        self._eval_after_training,
                        self._compute_badge_gradients,
                        self._train_button,
                        self._footer)
        
        figures = pn.Column(pn.pane.Markdown("### Training Loss"),
                            self._loss_fig,
                            pn.pane.Markdown("### Outputs By Class"),
                            self._hist_selector,
                            self._hist_fig)
        return pn.Row(controls, figures)
    
    
    def _train_callback(self, *event):
        # update the training function
        self.pw.build_training_step(self._learn_rate.value)
        self.pw._model_params["training"] = {"learn_rate":self._learn_rate.value,
                                             "batch_size":self._batch_size.value,
                                             "samples_per_epoch":self._samples_per_epoch.value,
                                             "epochs":self._epochs.value}
        # for each epoch
        epochs = self._epochs.value
        for e in range(epochs):
            self._footer.object = "### TRAININ (%s / %s)"%(e+1, epochs)
            self.pw._run_one_training_epoch(self._batch_size.value,
                                            self._samples_per_epoch.value)
            self.loss = self.pw.training_loss
            
        if self._eval_after_training.value:
            self._footer.object = "### EVALUATING"
            self.pw.predict_on_all(self._batch_size.value)
            self.pw._mlflow_track_run()
            
        if self._compute_badge_gradients.value:
            self._footer.object = "### COMPUTING BADGE GRADIENTS"
            self.pw.compute_badge_embeddings()
        
        self._loss_fig.object = _loss_fig(self.loss)
        self._hist_callback()
        self._footer.object = "### DONE"
        self.pw.save()
        
        
    def _hist_callback(self, *events):
        self._hist_fig.object = _hist_fig(self.pw.df, 
                                          self.pw.pred_df,
                                          self._hist_selector.value)
        

            

        
        
        






