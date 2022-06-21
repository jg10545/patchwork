"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

from patchwork._sample import _prepare_df_for_stratified_sampling




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
    
    

def _auc_and_acc(pos, neg, rnd=3):
    """
    macro for computing ROC AUC score and accuracy for display from arrays
    of scores for positive and negative cases
    """
    if (len(pos) > 0)&(len(neg) > 0):
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        y_pred = np.concatenate([pos, neg])
        return round(roc_auc_score(y_true, y_pred), rnd), round(accuracy_score(y_true,
                                                            (y_pred >= 0.5).astype(int)), rnd)
    else:
        return 0, 0


def _empty_fig():
    """
    make an empty matplotlib figure
    """
    fig, ax = plt.subplots()
    ax.plot([])
    ax.axis("off")
    plt.close(fig)
    return fig

def _loss_fig(l, ss_loss, testloss=None, testlossstep=None):
    """
    Generate matplotlib figure for a line plot of the 
    training loss
    """
    fig, ax = plt.subplots()
    ax.plot(l, "-", label="supervised")
    ax.plot(ss_loss, "--", label="semisupervised")
    if (testloss is not None)&(testlossstep is not None):
        ax.plot(testlossstep, testloss, "o-", label="validation")
    ax.set_xlabel("step", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)
    ax.grid(True)
    ax.legend(loc="upper right")
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
    train_auc, train_acc = _auc_and_acc(pos_labeled, neg_labeled)
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
    test_auc, test_acc = _auc_and_acc(pos_labeled, neg_labeled)
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
        a.set_ylabel("frequency", fontsize=14)
    ax1.set_xticks([])
    ax2.set_xlabel("model output", fontsize=14)
    title = "model outputs for '%s'\ntraining AUC  %s, accuracy %s"%(c, train_auc, train_acc)
    ax1.set_title(title, fontsize=14)
    
    title = "test AUC  %s, accuracy %s"%(test_auc, test_acc)
    ax2.set_title(title, fontsize=14)
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
        self._abort = False
        self.pw = pw
        self._header = pn.pane.Markdown("#### Train model on current set of labeled patches")
        self._sample_chooser = pn.widgets.Select(name="Sampling strategy", 
                                                 options=["class", "instance", "squareroot"],
                                                 value="class")
        self._opt_chooser = pn.widgets.Select(name="Optimizer", options=["adam", "momentum"],
                                              value="adam")
        self._learn_rate =  pn.widgets.LiteralInput(name="Initial learning rate", value=1e-3, type=float)                               
        self._lr_decay = pn.widgets.Select(name="Learning rate decay", options=["none", "cosine"],
                                             value="none")
        self._batch_size = pn.widgets.LiteralInput(name='Batch size', value=16, type=int)
        self._batches_per_epoch = pn.widgets.LiteralInput(name='Batches per epoch', value=100, type=int)
        self._epochs = pn.widgets.LiteralInput(name='Epochs', value=10, type=int)
        self._fine_tune_after = pn.widgets.LiteralInput(name="Fine-tune feature extractor after this many epochs", value=-1, type=int)
        self._pred_batch_size = pn.widgets.LiteralInput(name='Prediction batch size', value=64, type=int)
        self._abort_condition = pn.widgets.LiteralInput(name='Abort if no improvement in X epochs', value=0, type=int)
        
        self._eval_after_training = pn.widgets.Checkbox(name="Update predictions after training?", value=True)
        self._train_button = pn.widgets.Button(name="Make it so")
        self._train_button.on_click(self._train_callback)
        
        self._badge_chooser = pn.widgets.Select(name="Compute BADGE with respect to", 
                                                 options=["all classes"]+pw.classes,
                                                 value="all classes")
        self._compute_badge_gradients = pn.widgets.Button(name="Update BADGE gradients")
        self._compute_badge_gradients.on_click(self._badge_callback)
        
        self._footer = pn.pane.Markdown("")
        
        # objects to hold figures
        self._loss_fig = pn.pane.Matplotlib(_empty_fig(), width=500, height=300)
        self._hist_fig = pn.pane.Matplotlib(_empty_fig(), width=500, height=300)
        self._hist_selector = pn.widgets.Select(name="Class", options=self.pw.classes)
        self._hist_watcher = self._hist_selector.param.watch(
                        self._hist_callback, ["value"])
        
    def panel(self):
        """
        Return code for a training panel
        """
        controls =  pn.Column(self._header,
                        self._sample_chooser,
                        self._opt_chooser,
                        self._learn_rate,
                        self._lr_decay,
                         self._batch_size, 
                         self._batches_per_epoch,
                         self._epochs,
                         self._fine_tune_after,
                         self._pred_batch_size,
                         self._abort_condition,
                        self._eval_after_training,
                        self._train_button,
                        pn.layout.Divider(),
                        self._badge_chooser,
                        self._compute_badge_gradients,
                        pn.layout.Divider(),
                        self._footer)
        
        figures = pn.Column(pn.pane.Markdown("### Training Loss"),
                            self._loss_fig,
                            pn.pane.Markdown("### Outputs By Class"),
                            self._hist_selector,
                            self._hist_fig)
        return pn.Row(controls, figures)
    
    
    def _train_callback(self, *event):
        # check for abort condition
        abort = False
        epochs_since_improving = 0
        abort_condition = self._abort_condition.value
        
        # update the training function
        lr = self._learn_rate.value
        opt_type = self._opt_chooser.value
        if self._lr_decay.value == "cosine":
            # decay to 0 over the course of the experiment
            lr_decay = self._batch_size.value*self._batches_per_epoch.value*self._epochs.value
        else:
            lr_decay = 0
        self.pw.build_training_step(opt_type=opt_type, lr=lr, lr_decay=lr_decay, 
                                    decay_type="cosine", weight_decay=self.pw._model_params["weight_decay"])
        self.pw._model_params["training"] = {"learn_rate":self._learn_rate.value,
                                             "batch_size":self._batch_size.value,
                                             "batches_per_epoch":self._batches_per_epoch.value,
                                             "epochs":self._epochs.value,
                                             "sampling":self._sample_chooser.value,
                                             "fine_tune_after":self._fine_tune_after.value}
        
        # tensorflow function for computing test loss
        meanloss = self.pw._build_loss_tf_fn()
        val_ds = self.pw._val_dataset(self._batch_size.value)
        # precompute subsets of dataframe for each class/label for
        # stratified sampling
        indexlist = _prepare_df_for_stratified_sampling(self.pw.df)
        sampling = self._sample_chooser.value
        
        # for each epoch
        epochs = self._epochs.value
        fine_tune_after = self._fine_tune_after.value
        for e in range(epochs):
            # fine tuning case: rebuild the training step after the right
            # number of epochs
            if e == fine_tune_after:
                self._footer.object = "### UPDATING TRAINING STEP"
                self.pw.build_training_step(opt_type=opt_type, lr=lr, lr_decay=lr_decay, 
                                    decay_type="cosine",
                                    weight_decay=self.pw._model_params["weight_decay"],
                                    finetune=True)
            self._footer.object = "### TRAININ (%s / %s)"%(e+1, epochs)
            
            
            N = self._batch_size.value * self._batches_per_epoch.value
            self.pw._run_one_training_epoch(self._batch_size.value, N,
                                            sampling, indexlist)
            self.loss = self.pw.training_loss
            
            # at the end of each epoch compute test loss
            testloss = np.mean([meanloss(x,y).numpy() for x,y in val_ds])
            self.pw.test_loss.append(testloss)
            self.pw.test_loss_step.append(len(self.pw.training_loss))
            
            self._loss_fig.object = _loss_fig(self.loss,
                                          self.pw.semisup_loss,
                                          self.pw.test_loss,
                                          self.pw.test_loss_step)
            # CHECK TO SEE IF WE SHOULD ABORT
            if len(self.pw.test_loss) >= 2:
                # if loss went down this epoch
                if self.pw.test_loss[-1] < self.pw.test_loss[-2]:
                    epochs_since_improving = 0
                else:
                    epochs_since_improving += 1
                    
            if (abort_condition > 0)&(epochs_since_improving >= abort_condition):
                abort = True
                self._footer.object = f"### ABORTING AT EPOCH {e}"
                break
            
            
        if self._eval_after_training.value:
            # skip this if we're aborting
            if not abort:
                self._footer.object = "### EVALUATING"
                self.pw.predict_on_all(self._pred_batch_size.value)
                self.pw._mlflow_track_run()

        if not abort:
            self.pw._update_diversity_sampler()
            self._hist_callback()
            self.pw.save()
            self._footer.object = "### DONE"
        
    def _badge_callback(self, *events):
        # the BADGE chooser lets you compute BADGE embeddings for diversity 
        # sampling with respect to all categories or just one
        cat = self._badge_chooser.value
        if cat == "all classes":
            cat = None
        else:
            cat = self.pw.classes.index(cat)
        self._footer.object = "### COMPUTING BADGE GRADIENTS"
        self.pw.compute_badge_embeddings(cat)
        self._footer.object = "### DONE"
        
        
    def _hist_callback(self, *events):
        self._hist_fig.object = _hist_fig(self.pw.df, 
                                          self.pw.pred_df,
                                          self._hist_selector.value)
        
    def _abort_callback(self, *events):
        print("abortiiiing")
        self._abort = True
        

            

        
        
        






