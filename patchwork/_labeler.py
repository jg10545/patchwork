"""

                _modelpicker.py


GUI code for training a model

"""
import pandas as pd
import panel as pn


def _reformat(x):
    return str(x).split("\nName")[0].replace("\n", "<br>")

def single_image_chooser(classes, imsize=100):
    img = pn.panel("default_img.gif", width=imsize, height=imsize)
    class_choice = pn.widgets.RadioBoxGroup(name="Class", options=classes)
    return pn.Row(img, class_choice)


def pick_indices(df, M, sort_by, labeled=True):
    """
    Function to handle selecting indices of images to label
    """
    if labeled:
        subset = df[pd.notnull(df["label"])]
    else:
        subset = df[pd.isnull(df["label"])]
            
    #if len(subset) > 0:
    M = min(len(subset), M)
    # RANDOM SAMPLING
    if sort_by == "random":
        sample = subset.sample(M)
        indices = sample.index.to_numpy()
    # UNCERTAINTY SAMPLING
    elif sort_by == "max entropy":
        if "entropy" in df.columns:
            subset = subset["entropy"].nlargest(M)
            indices = subset.index.to_numpy()
    return indices



class Labeler():
    """
    Class to manage displaying images and gathering user feedback
    """
    
    def __init__(self, classes, df, dim=2, imsize=100):
        self._classes = classes
        self._label_types = [None, "(exclude)"] + list(classes)
        self._dim = dim
        self._df = df
        self._indices = "nuthin here"
        
        self._choosers = [single_image_chooser(self._label_types, imsize)
                          for _ in range(dim**2)]
        
        self.GridSpec = pn.GridSpec(sizing_mode="stretch_both",
                                 max_width=50)
        i = 0
        for r in range(dim):
            for c in range(dim):
                self.GridSpec[r,c] = self._choosers[i]
                i += 1
              
        self._build_select_controls()
        
    def _fill_images(self, indices):
        """
        code to update the images and labels in the 
        annotation tab
        """
        self._indices = indices
        
        for i, c in enumerate(self._choosers):
            if i < len(indices):
                c[0].object = self._df["filepath"].iloc[indices[i]]
                c[1].value = self._df["label"].iloc[indices[i]]
            else:
                c[0].object = "default_img.gif"
                c[1].value = None

        
    def panel(self):
        """
        Return code for an annotation panel
        """
        return pn.Column(self._select_controls, 
                         self.GridSpec,
                        self._label_save_controls)
    
    def _build_select_controls(self):
        """
        Generate all the widgets to sample images and output results
        """
        opts= ["random", "max entropy"]
        self._sort_by = pn.widgets.Select(name="Sort by", options=opts)
        self._retrieve_button = pn.widgets.Button(name="Retrieve")
        self._retrieve_watcher = self._retrieve_button.param.watch(
                        self._retrieve_callback, ["clicks"])
        self._whether_labeled_radio = pn.widgets.RadioBoxGroup(name="labeled", 
                                                options=["unlabeled", "labeled"])
        
        self._update_label_button = pn.widgets.Button(name="Update Labels")
        self._update_watcher = self._update_label_button.param.watch(
                        self._update_label_callback, ["clicks"])
        self._label_counts = pn.pane.Markdown("")
        self._select_controls = pn.Row(
            self._sort_by, 
            self._whether_labeled_radio,
            self._retrieve_button
            
        )
        self._label_save_controls = pn.Row(
            self._update_label_button, self._label_counts
        )
        
    def _retrieve_callback(self, *events):
        """
        Callback to sample images for labeling
        """
        sort_by = self._sort_by.value
        labeled = self._whether_labeled_radio.value == "labeled"
        
        indices = pick_indices(self._df, self._dim**2, 
                               sort_by, labeled)
        if len(indices) > 0:
            self._fill_images(indices)
            
    def _update_label_callback(self, *events):
        labels = self._df["label"].values
        for i, ind in enumerate(self._indices):
            newval = self._choosers[i][1].value
            labels[ind] = newval
        self._df["label"] = labels
        
        self._label_counts.object = _reformat(self._df["label"].value_counts())