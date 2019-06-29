"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import panel as pn
import tensorflow as tf



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
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("cross-entropy", fontsize=14)
    ax.grid(True)
    plt.close(fig)
    return fig


def _hist_fig(df, pred, c):
    """
    
    """
    pos_labeled = pred[c][df[c] == 1].values
    neg_labeled = pred[c][df[c] == 0].values
    unlabeled = pred[c][pd.isnull(df[c])].values
    bins = np.linspace(0, 1, 15)

    fig, ax = plt.subplots()
    ax.hist(pos_labeled, bins=bins, alpha=0.5, label="labeled positive", normed=True)
    ax.hist(neg_labeled, bins=bins, alpha=0.5, label="labeled negative", normed=True)
    ax.hist(unlabeled, bins=bins, alpha=0.5, label="unlabeled", normed=True)
    ax.legend(loc="upper left")
    ax.set_xlabel("assessed probability", fontsize=14)
    ax.set_ylabel("frequency", fontsize=14)
    ax.set_title("sigmoid outputs for '%s'"%c, fontsize=14)
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
        self._batch_size = pn.widgets.LiteralInput(name='Batch size', value=16, type=int)
        self._samples_per_epoch = pn.widgets.LiteralInput(name='Samples per epoch', value=1000, type=int)
        self._epochs = pn.widgets.LiteralInput(name='Epochs', value=10, type=int)
        
        self._eval_after_training = pn.widgets.Checkbox(name="Update predictions after training?", value=True)
        self._train_button = pn.widgets.Button(name="Make it so")
        self._train_button_watcher = self._train_button.param.watch(
                        self._train_callback, ["clicks"])
        
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
                         self._batch_size, 
                         self._samples_per_epoch,
                         self._epochs,
                        self._eval_after_training,
                        self._train_button,
                        self._footer)
        
        figures = pn.Column(pn.pane.Markdown("### Training Loss"),
                            self._loss_fig,
                            pn.pane.Markdown("### Outputs By Class"),
                            self._hist_selector,
                            self._hist_fig)
        return pn.Row(controls, figures)
    
    
    def _train_callback(self, *event):
        # for each epoch
        epochs = self._epochs.value
        self.loss = []
        for e in range(epochs):
            self._footer.object = "### TRAININ (%s / %s)"%(e+1, epochs)
            history = self.pw.fit(self._batch_size.value, self._samples_per_epoch.value)
            self.loss.append(history.history["loss"][-1])
            
        if self._eval_after_training.value:
            self._footer.object = "### EVALUATING"
            self.pw.predict_on_all(self._batch_size.value)
            
        self._loss_fig.object = _loss_fig(self.loss)
        self._footer.object = "### DONE"
        
        
    def _hist_callback(self, *events):
        self._hist_fig.object = _hist_fig(self.pw.df, 
                                          self.pw.pred_df,
                                          self._hist_selector.value)
            
    def _train_callback_orig(self, *event):
        # update the model in the Patchwork object
        self.pw.model = self.pw.modelpicker.model
        # compile the model
        self.pw.model.compile(tf.keras.optimizers.RMSprop(1e-3),
                   loss=tf.keras.losses.sparse_categorical_crossentropy)
        # initialize a training generator
        gen = self.pw._training_generator(self._batch_size.value)
        
        loss = []
        epochs = self._epochs.value
        for e in range(epochs):
            self._footer.object = "### TRAININ (%s / %s)"%(e+1, epochs)
            history = self.pw.model.fit_generator(gen,
                                    steps_per_epoch=self._training_steps.value,
                                    epochs=1)
            loss.append(history.history["loss"][-1])
        # update evaluations
        if self._eval_after_training.value:
            self._footer.object = "### EVALUATING"
            preds = self.pw.model.predict(self.pw.feature_vecs)
            
            for i, c in enumerate(self.pw.classes):
                self.pw.df[c] = preds[:,i]
            
            self.pw.df["entropy"] = shannon_entropy(preds)

        self.loss = loss
        self._footer.object = "### DONE"
        
        
        
        





