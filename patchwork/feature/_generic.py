"""

            _generic.py

I feel like I'm rewriting a lot of the same code- so let's define a master
class holding all the stuff we want to reuse between feature extractors.

"""
import os
import yaml

import tensorflow as tf
from tqdm import tqdm


INPUT_PARAMS = ["imshape", "num_channels", "norm", "batch_size",
                "shuffle", "num_parallel_calls"]



class GenericExtractor(object):
    """
    Place to store common code for different feature extractor methods.
    
    Don't actually use this to do anything.
    """
    
    
    def __init__(self, logdir, trainingdata, fcn=None, augment=False, 
                 extractor_param=None, imshape=(256,256), num_channels=3,
                 norm=255, batch_size=64, shuffle=True, num_parallel_calls=None):
        """
        
        """
        self.logdir = logdir
        
        if fcn is None:
            fcn = self._build_default_model()
        self.fcn = fcn
        self._models = {"fcn":fcn}
        
        self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
        self._file_writer.set_as_default()
        self.step = 0
        
        
        
    def _parse_configs(self, **kwargs):
        """
        Organize input parameters and save to a YAML file so you can
        find them later.
        """
        self.config = {}
        self.input_config = {}
        self.augment_config = False
        
        for k in kwargs:
            if k == "augment":
                self.augment_config = kwargs[k]
            elif k in INPUT_PARAMS:
                self.input_config[k] = kwargs[k]
            else:
                self.config[k] = kwargs[k]
                
        config_path = os.path.join(self.logdir, "config.yml")
        config_dict = {"model":self.config, "input":self.input_config, 
                       "augment":self.augment}
        yaml.dump(config_dict, open(config_path, "w"), default_flow_style=False)
        
        
    def _build_default_model(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def _run_training_epoch(self, **kwargs):
        # REPLACE THIS WHEN SUBCLASSING
        return True
    
    def fit(self, epochs=1, save=True, evaluate=True):
        """
        Train the feature extractor
        
        :epochs: number of epochs to train for
        :save: if True, save after each epoch
        :evaluate: if True, run eval metrics after each epoch
        """
        for e in tqdm(range(epochs)):
            self._run_training_epoch()
            
            if save:
                self.save()
            if evaluate:
                self.evaluate()
            self.step += 1
    
    def save(self):
        """
        Write model(s) to disk
        """
        for m in self._models:
            path = os.path.join(self.logdir, m)
            self._models[m].save(path, overwrite=True)
            
    def evaluate(self):
        # REPLACE THIS WHEN SUBCLASSING
        return True
            
    def _record_scalars(self, **scalars):
        for s in scalars:
            tf.summary.scalar(s, scalars[s], step=self.step)
        
            
    