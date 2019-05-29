"""

                _modelpicker.py


GUI code for choosing a model

"""
import param
import panel as pn
from patchwork._models import model_list


class ModelPicker(param.Parameterized):
    """
    
    """
    model = "no model yet"
    num_classes = param.Integer(default=2, bounds=(2,100),
                               label="Number of classes", constant=True)
    inpt_channels = param.Integer(default=512, bounds=(1,4096), constant=True,
                               label="Dimension of pretrained embeddings")
    model_type = param.ObjectSelector(default=model_list[0], 
                                objects=model_list)
    
    build_model = param.Action(lambda x: x._build_model(),
                              label="Build Model",
                              doc="Create a new model with these settings")
    
    def panel(self):
        return pn.panel(self, expand=True, expand_button=False)
    
    
    def _build_model(self):
        self.model = self.model_type._build(self.num_classes, self.inpt_channels)
        