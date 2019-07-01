"""

                _modelpicker.py


GUI code for choosing a model

"""
import param
import panel as pn
from patchwork._models import model_dict


class _ModelPicker(param.Parameterized):
    """
    DEPRECATED VERSION
    
    For building fine-tuning models
    """
    model = "no model yet"
    num_classes = param.Integer(default=2, bounds=(1,100),
                               label="Number of classes", constant=True)
    inpt_channels = param.Integer(default=512, bounds=(1,4096), constant=True,
                               label="Dimension of input embeddings")
    model_type = param.ObjectSelector(default=model_dict["Linear"], 
                                objects=model_dict)
    
    build_model = param.Action(lambda x: x._build_model(),
                              label="Build Model",
                              doc="Create a new model with these settings")
    
    def panel(self):
        return pn.panel(self, expand=True, expand_button=False)
    
    
    def _build_model(self):
        self.model = self.model_type._build(self.num_classes, self.inpt_channels)
        
   

class ModelPicker(object):
    """
    For building fine-tuning models
    """
    def __init__(self, num_classes, inpt_channels, pw):
        """
        
        """
        self._current_model = pn.pane.Markdown("**Current fine-tuning model:** None")
        self._num_classes = num_classes
        self._inpt_channels = inpt_channels
        self._pw = pw
        # dropdown and watcher to pick model type
        self._model_chooser = pn.widgets.Select(options=model_dict)
        self._chooser_watcher = self._model_chooser.param.watch(self._chooser_callback, 
                                                                 ["value"])
        # model description pane
        self._model_desc = pn.pane.Markdown(self._model_chooser.value.description)
        # model hyperparameters
        self._hyperparams = pn.panel(self._model_chooser.value)
        
        # model-builder button
        self._build_button = pn.widgets.Button(name="Build Model")
        self._build_button.on_click(self._build_callback)
        
        # semi-supervised
        self._entropy_reg = pn.widgets.LiteralInput(name='Entropy Regularization Weight', 
                                                    value=0., type=float)
        
    def panel(self):
        model_options = pn.Column(
            pn.pane.Markdown("### Model Options"),
            self._current_model,
            self._model_chooser,
            self._model_desc,
            self._hyperparams,
            self._build_button
        )
        
        semi_supervized_options = pn.Column(
                pn.pane.Markdown("### Semi-Supervised Learning\n\n*rebuild model after updating*"),
                self._entropy_reg
                )
        
        return pn.Row(model_options, semi_supervized_options)
    
    def _chooser_callback(self, *events):
        # update description and hyperparameters
        self._model_desc.object = self._model_chooser.value.description
        if hasattr(pn.panel(self._model_chooser.value), "objects"):
            self._hyperparams.objects = pn.panel(self._model_chooser.value).objects
        else:
            
            self._hyperparams.objects = []
        
    def _build_callback(self, *events):
        self._pw.fine_tuning_model = self._model_chooser.value._build(self._num_classes, 
                                                                      self._inpt_channels)
        self._pw.build_model(entropy_reg=self._entropy_reg.value)
        self._current_model.object = "**Current fine-tuning model:** %s"%self._model_chooser.value.name
        self._pw.trainmanager.loss = []

     
# NEW VERSION: I think I might have to abandon param. not sure how the objectselector
# would work though.
        

#def assemble_model(num_classes, inpt_shape, feature_extractor=None):