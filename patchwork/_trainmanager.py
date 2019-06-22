"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import panel as pn
import tensorflow as tf



#def shannon_entropy(x):
#    """
#    Shannon entropy of a 2D array
#    """
#    xprime = np.maximum(np.minimum(x, 1-1e-8), 1e-8)
#    return -np.sum(xprime*np.log2(xprime), axis=1)



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
        self._training_steps = pn.widgets.LiteralInput(name='Steps per epoch DEPRECATED', value=100, type=int)
        self._epochs = pn.widgets.LiteralInput(name='Epochs', value=10, type=int)
        
        self._eval_after_training = pn.widgets.Checkbox(name="Update predictions after training?", value=True)
        self._train_button = pn.widgets.Button(name="Make it so")
        self._train_button_watcher = self._train_button.param.watch(
                        self._train_callback, ["clicks"])
        
        self._footer = pn.pane.Markdown("")
        
    def panel(self):
        """
        Return code for a training panel
        """
        return pn.Column(self._header,
                         self._batch_size, 
                         self._training_steps,
                         self._epochs,
                        self._eval_after_training,
                        self._train_button,
                        self._footer)
    
    
    def _train_callback(self, *event):
        # for each epoch
        epochs = self._epochs.value
        self.loss = []
        for e in range(epochs):
            self._footer.object = "### TRAININ (%s / %s)"%(e+1, epochs)
            history = self.pw.fit(self._batch_size.value)
            self.loss.append(history.history["loss"][-1])
            
        if self._eval_after_training.value:
            self._footer.object = "### EVALUATING"
            self.pw.predict_on_all(self._batch_size.value)
            
        
        self._footer.object = "### DONE"
        
        
            
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
        
        
        
        






