import numpy as np
import tensorflow as tf
import panel as pn

from patchwork.viz._validation_scatter import _get_tsne_features, _build_scatter_holomap


class ErrorVisualizer():
    
    def __init__(self, pw):
        self._pw = pw
        self.classes = pw.classes
        self.features = np.zeros((2,2))
        
        # SET UP WIDGETS FOR GUI
        self._update_button = pn.widgets.Button(name="Update")
        self._cat_chooser = pn.widgets.Select(options=self.classes)
        
        self._holo = pn.pane.HoloViews()
        
        # LINK BUTTONS
        self._update_button.on_click(self._update_callback)
        self._cat_watcher = self._cat_chooser.param.watch(
                        self._select_callback, ["value"])
        
    def panel(self):
        return pn.Column(
            pn.Row(self._update_button,
                   self._cat_chooser),
                              self._holo)
        
    def _update_features(self):
        ds = self._pw._val_dataset()
        
        fcn = self._pw.models["feature_extractor"]
        inpt = tf.keras.layers.Input(fcn.input_shape[1:])
        net = fcn(inpt)
        net = self._pw.models["fine_tuning"](net)
        model = tf.keras.Model(inpt, net)
        
        features = model.predict(ds)
        self._embeds = _get_tsne_features(features)
        
        
    def _update_callback(self, *events):
        self._update_features()
        
        val = self._pw.df.validation
        val_df = self.pw.df[val]
        val_pred_df = self.pw.pred_df[val]
        
        assert self._embeds.shape[0] == len(val_df), "WRONG NUMBER OF VALIDATION EXAMPLES???"
        self._figures = _build_scatter_holomap(val_df, val_pred_df, self._embeds, return_dict=True)
        self._select_callback()
    
    def _select_callback(self, *events):
        self._holo.object = self._figures[self._cat_chooser.value]