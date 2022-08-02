"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import panel as pn
import os

import warnings



from patchwork._sample import find_subset
from patchwork._util import shannon_entropy




def _gen_figs(arrays, dim=3, lw=5):
    """
    Generate a list of matplotlib figures for each case of
    highlighted image
    
    :arrays: list of numpy arrays of images
    :dim: grid dimension
    :lw: width of highlight rectangles
    """
    figs = []
    assert len(figs) <= dim*dim, "too many arrays"
    
    for i in range(len(arrays)):
        fig, ax = plt.subplots(dim, dim)
        fig.set_figwidth(6)
        fig.set_figheight(6)
        ax = ax.ravel()
        
        for j in range(dim**2):
            if j < len(arrays):
                ax[j].imshow(arrays[j], aspect="equal")
                if i == j:
                    a = ax[j].axis()
                    rect = Rectangle((lw,lw),a[1]-2*lw,a[2]-2*lw,
                                     linewidth=lw,edgecolor='r',facecolor='none')
                    ax[j].add_patch(rect)
            ax[j].axis(False)
        plt.tight_layout()
        plt.close(fig)
        figs.append(fig)
    return figs



def _single_class_radiobuttons(width=125, height=25):
    """
    Generate buttons for a single class label: None, 0, and 1
    """
    return pn.widgets.RadioButtonGroup(options=["None", "0", "1"], width=width, align="center", 
                                       #height=height, value="None")
                                       value="None")




class ButtonPanel(object):
    """
    panel widget that has all the pieces for displaying and labeling patches
    """
    def __init__(self, classes, df, load_func, dim=3, size=700):
        """
        :classes: list of strings- name of each class
        :df: pandas DataFrame
        :load_func: function to load a file from a path to an array
        :dim: dimensions of display grid
        """
        self._index = None
        self._classes = classes
        self._df = df
        self._load_func = load_func
        
        self.dim = dim
        self._num_images = dim**2

        self._figpanel = pn.pane.Matplotlib(height_policy="fit",
                                            width_policy="fit",
                                            aspect_ratio=1,
                                            width=size, height=size)
        
        self._value_map = {'None': np.nan, '0': 0., '1': 1., None:np.nan,
                           np.nan:np.nan}

        self._selections = {c:_single_class_radiobuttons() for c in classes}
        self._exclude = pn.widgets.Checkbox(name="exclude", align="center")
        self._validation = pn.widgets.Checkbox(name="validation", align="center")
      
        self._back_button = pn.widgets.Button(name='\u25c0', width=50)
        self._back_button.on_click(self._back_button_callback)
        self._forward_button = pn.widgets.Button(name='\u25b6', width=50)
        self._forward_button.on_click(self._forward_button_callback)
        
        self._zero_all_button = pn.widgets.Button(name="all zeros", width=100)
        self._zero_all_button.on_click(self._zero_all_callback)
        
        self._copy_labels_button = pn.widgets.Button(name="copy labels to all", width=100)
        self._copy_labels_button.on_click(self._copy_labels_callback)
        
        self.label_counts = pn.pane.Markdown("")
        
        self._button_panel = pn.Column(pn.Spacer(height=50),
                                       pn.pane.Markdown("## Image labels"),
                                        self._exclude, 
                                        self._validation,
                                      *[pn.Row(self._selections[c], 
                                            pn.pane.Markdown(c, align="center"), 
                                            height=30)
                                     for c in classes],
                                      pn.Spacer(height=10),
                                     pn.Row(self._back_button, self._forward_button),
                                     self._zero_all_button,
                                     self._copy_labels_button,
                                      self.label_counts) 
        
        
        
        self._selected_image = 0
        self._figpanel.select(0)
        self.panel = pn.Row(self._figpanel, pn.Spacer(width=50), self._button_panel)
        self._indices = None
        
    def load(self, indices):
        """
        Input an array of indices and load them
        
        :indices: array of indices to load from dataframe
        """
        # pull out the filepaths associated with indices
        assert len(indices) <= self._num_images, "Too many indices"
        if self._indices is not None:
            self.record_values()
        self._indices = indices
        filepaths = self._df["filepath"].iloc[indices]
        # load to numpy arrays
        self._image_arrays = [self._load_func(f) for f in filepaths]
        # make a list of matplotlib figures, one for each image that
        # could be highlighted
        self._figs = _gen_figs(self._image_arrays, dim=self.dim, lw=5)
        # highlight first image
        self._figpanel.object = self._figs[0]
        self._selected_image = 0
        self._update_buttons()
        
        
    def _update_buttons(self):
        """
        Update buttons for current selection
        """
        record = self._df.iloc[self._indices[self._selected_image]]
        self._exclude.value = bool(record["exclude"])
        self._validation.value = bool(record["validation"])
        
        for c in self._classes:
            if np.isnan(record[c]):
                self._selections[c].value = "None"
            else:
                self._selections[c].value = str(int(record[c]))
                
                
    def _zero_all_callback(self, *events):
        """
        Update labels for the current selection so they're all zero
        """
        for c in self._classes:
            self._selections[c].value = "0"
        self._forward_button_callback()
        
    def _copy_labels_callback(self, *events):
        """
        Take your current labels and copy them to
        all the images currently viewable
        """
        labels = {c:self._selections[c].value for c in self._classes}
        for j in range(self.dim**2):
            i = self._indices[j]
            for c in self._classes:
                self._df.loc[i,c] = self._value_map[labels[c]]
        
    def select(self, i):
        """
        Select the ith displayed image
        """
        assert i >= 0
        assert i < self.dim**2
        
        self._figpanel.object = self._figs[i]
        self._selected_image = i
        self._update_buttons()
    
    def record_values(self):
        """
        Save current button values to dataframe
        """
        i = self._indices[self._selected_image]
        self._df.loc[i, "exclude"] = self._exclude.value
        self._df.loc[i, "validation"] = self._validation.value
        for c in self._classes:
            self._df.loc[i,c] = self._value_map[self._selections[c].value]
    
    def _forward_button_callback(self, *events):
        self.record_values()
        if self._selected_image < self._num_images - 1:
            self._selected_image += 1
            self.select(self._selected_image)
        
    def _back_button_callback(self, *events):
        self.record_values()
        if self._selected_image > 0:
            self._selected_image -= 1
            self.select(self._selected_image)


def _generate_label_summary(df, classes):
    text = "\n### Label Counts\n"
    
    #text += "#### training\n"
    text +=  "\n| training | neg | pos | \n| -----|  ----- | ----- | \n"
    for c in classes:
        text += "| %s | %s | %s | \n"%(c, (df[~df["validation"]][c]==0).sum(),
                                      (df[~df["validation"]][c]==1).sum()) 
    #text += "\n \n#### validation\n "
    text +=  "\n| validation | neg | pos | \n| -----|  ----- | ----- |\n"
    for c in classes:
        text += "| %s | %s | %s | \n"%(c, (df[df["validation"]][c]==0).sum(),
                                      (df[df["validation"]][c]==1).sum())
        
    return text




def pick_indices(df, pred_df, M, label_status, exclude_status,
                 sort_by, subset_by, sampler=None, divsampler=None):
    """
    Function to handle selecting indices of images to label
    
    :df: dataframe containing labels
    :pred_df: dataframe containing current model predictions
    :M: number of indices to retrieve
    :label_status: "unlabeled", "partial", or "labeled"
    :exclude_status: "not excluded", "excluded", or "validation"
    :sort_by: method for ordering results
    :subset_by: method for subsetting df before ordering
    :sampler: optional; custom sampling object for BADGE
    """
    # take a subset of the data to sort through
    subset = find_subset(df, label_status, exclude_status, subset_by)
    if subset.sum() == 0:
        warnings.warn("empty subset")
    df = df[subset]
    pred_df = pred_df[subset]

    M = min(len(df), M)
    # RANDOM SAMPLING
    if sort_by == "random":
        sample = df.sample(M)
    # UNCERTAINTY SAMPLING
    elif sort_by == "max entropy":
        pred_df = pred_df.assign(entropy=shannon_entropy(pred_df.values))
        sample = pred_df["entropy"].nlargest(M)
    # SINGLE-CLASS UNCERTAINTY SAMPLING
    elif "maxent:" in sort_by:
        col = sort_by.replace("maxent:","").strip()
        pred_df = pred_df.assign(entropy=shannon_entropy(pred_df[[col]].values))
        sample = pred_df["entropy"].nlargest(M)
    # FINDING HIGHEST/LOWEST CASES DETERMINISTICALLY
    elif "highest:" in sort_by:
        col = sort_by.replace("highest: ", "")
        sample = pred_df[col].nlargest(M)
    elif "lowest:" in sort_by:
        col = sort_by.replace("lowest: ", "")
        sample = pred_df[col].nsmallest(M)
    # FINDING HIGHEST/LOWEST CASES STOCHASTICALLY
    elif "high:" in sort_by:
        col = sort_by.replace("high: ", "")
        # scale sampling probability by score
        p = pred_df[col].values**4
        sample = pred_df[col].sample(M, weights=p)
    elif "low:" in sort_by:
        col = sort_by.replace("low: ", "")
        p = (1-pred_df[col].values)**4
        sample = pred_df[col].sample(M, weights=p)
    elif ("BADGE" in sort_by)&(sampler is not None):
        indices = np.array(sampler(M, include=subset.values))
        return indices
    elif ("diversity" in sort_by)&(divsampler is not None):
        indices = np.array(divsampler(M, include=subset.values))
        return indices
    
    indices = sample.index.to_numpy()
    return indices



class Labeler():
    """
    Class to manage displaying images and gathering user feedback
    """
    
    def __init__(self, classes, df, pred_df, load_func, pw, dim=3, 
                 logdir=None):
        """
        :classes: list of strings; class labels
        :df: DataFrame containing filepaths and class labels
        :pred_df: dataframe of current model predictions
        :load_func:
        :pw: patchwork.GUI object that calls the labeler
        :dim: dimension of the image grid to display
        :logdir: path to save labels to
        """
        self._classes = classes
        if "subset" in df.columns:
            self._subset_types = df["subset"].unique()
        self._dim = dim
        self._df = df
        self._pred_df = pred_df
        self._logdir = logdir
        self._buttonpanel = ButtonPanel(classes, df, load_func, dim)
        self._load_func = load_func
        self._pw = pw
             
        self._build_select_controls()

        
    def panel(self):
        """
        Return code for an annotation panel
        """
        return pn.Column(self._select_controls, 
                         self._buttonpanel.panel)
    
    def _build_select_controls(self):
        """
        Generate all the widgets to sample images and output results
        """
        # generate all sorting options
        sort_opts= ["random", "max entropy", "BADGE"]
        if self._pw._diversity_sampler is not None:
            sort_opts.append("diversity")
        for e in ["highest: ", "high: ", "lowest: ", "low: ", "maxent: "]:
            for c in self._classes:
                sort_opts.append(e+c)
                
        # generate all subsetting options
        #subset_opts = ["unlabeled", "fully labeled", "partially labeled", 
        #               "excluded", "not excluded", "validation"]
        subset_opts = ["all"]
        for e in ["unlabeled: ", "contains: ", "doesn't contain: "]:
            for c in self._classes:
                subset_opts.append(e+c)
                
        if hasattr(self, "_subset_types"):
            for s in self._subset_types:
                subset_opts.append(f"subset: {s}")
            
        
            
        self._sort_by = pn.widgets.Select(name="Sort by", options=sort_opts,
                                          value="random")
        self._retrieve_button = pn.widgets.Button(name="Sample", align="end")
        self._retrieve_button.on_click(self._retrieve_callback)
        self._subset_by = pn.widgets.Select(name="Subset by", options=subset_opts,
                                            value="unlabeled")
        self._label_status = pn.widgets.RadioBoxGroup(options=["unlabeled", "partial", "labeled"], inline=False, width=100, value="unlabeled")
        self._exclude_status = pn.widgets.RadioBoxGroup(options=["not excluded", "excluded", "validation"], inline=False, width=100, value="not excluded")
        
        self._update_label_button = pn.widgets.Button(name="Update Labels")
        self._label_counts = pn.pane.Markdown("")
        self._select_controls = pn.Row(
            self._label_status,
            self._exclude_status,
            self._subset_by, 
            self._sort_by,
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
        subset_by = self._subset_by.value
        label_status = self._label_status.value
        exclude_status = self._exclude_status.value
        indices = pick_indices(self._df, self._pred_df, self._dim**2, 
                               label_status, exclude_status,
                               sort_by, subset_by, self._pw._badge_sampler,
                               self._pw._diversity_sampler)
        if len(indices) > 0:
            self._buttonpanel.load(indices)
            self._buttonpanel.label_counts.object = _generate_label_summary(self._df, self._classes)
            if self._logdir is not None:
                self._df.to_csv(os.path.join(self._logdir, "labels.csv"), index=False)
        else:
            warnings.warn("no matching images to load")
        
 