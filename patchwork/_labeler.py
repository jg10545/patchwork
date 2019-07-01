"""

                _modelpicker.py


GUI code for training a model

"""
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#import pandas as pd
import panel as pn
from PIL import Image


from patchwork._sample import find_subset
from patchwork._util import shannon_entropy, tiff_to_array


def _load_to_fig(f, figsize=(5,5), lw=5):
    """
    :f: string; path to file
    """
    if ".tif" in f:
        im = tiff_to_array(f, channels=3)
    else:
        im = Image.open(f)
    
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.imshow(im)
    ax1.axis("off")
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.imshow(im)
    ax2.axis("off")
    a = ax2.axis()
    rect = Rectangle((lw,lw),a[1]-2*lw,a[2]-2*lw,
                     linewidth=lw,edgecolor='r',facecolor='none')
    ax2.add_patch(rect)
    plt.close(fig2)
    
    return fig1, fig2



class SingleImgDisplayer(object):
    """
    Widget to handle displaying a single image- precomputes matplotlib
    figures for selected and unselected cases
    """
    
    def __init__(self):
        fig, ax = plt.subplots()
        ax.plot([])
        ax.axis("off")
        self._fig_selected = fig
        self._fig_unselected = fig
        self.selected = False
        self.panel = pn.pane.Matplotlib(self._fig_unselected)#
        
    def load(self, filepath):
        """
        
        """
        self._fig_unselected, self._fig_selected = _load_to_fig(filepath)
        self.panel.object = self._fig_unselected
        
    def select(self):
        if not self.selected:
            self.panel.object = self._fig_selected
            self.selected = True

    def unselect(self):
        if self.selected:
            self.panel.object = self._fig_unselected
            self.selected = False


def _single_class_radiobuttons(width=125, height=25):
    """
    Generate buttons for a single class label: None, 0, and 1
    """
    return pn.widgets.RadioButtonGroup(options=["None", "0", "1"], width=width, align="center", 
                                       height=height, value="None")

class ButtonPanel(object):
    """
    panel widget that has all the pieces for displaying and labeling patches
    """
    def __init__(self, classes, df, dim=3):
        self._index = None
        self._classes = classes
        self._df = df
        
        self.dim = dim
        self._num_images = dim**2
        self._single_img_patches = [SingleImgDisplayer() for _ in range(dim**2)]
        self._figpanel = pn.GridSpec()
        
        k = 0
        for j in range(dim):
            for i in range(dim):
                self._figpanel[j,i] = self._single_img_patches[k].panel
                k += 1
        
        
        self._value_map = {"None":None, "0":0, "1":1}
        self._selections = {c:_single_class_radiobuttons() for c in classes}
        self._exclude = pn.widgets.Checkbox(name="exclude", align="center")
      
        self._back_button = pn.widgets.Button(name='\u25c0', width=50)
        self._back_button.on_click(self._back_button_callback)
        self._forward_button = pn.widgets.Button(name='\u25b6', width=50)
        self._forward_button.on_click(self._forward_button_callback)
        self.label_counts = pn.pane.Markdown("")
        
        self._button_panel = pn.Column(pn.Spacer(height=50),
                                       pn.pane.Markdown("## Image labels"),
                                        self._exclude, 
                                      *[pn.Row(self._selections[c], 
                                            pn.pane.Markdown(c, align="center"), 
                                            height=30)
                                     for c in classes],
                                     pn.Row(self._back_button, self._forward_button),
                                      self.label_counts) 
        
        
        
        self._selected_image = 0
        self._figpanel.select(0)
        self.panel = pn.Row(self._figpanel, pn.Spacer(width=50), self._button_panel)
        self._indices = None
        
    def load(self, indices, select_first=True):
        """
        Input an array of indices and load them
        """
        assert len(indices) <= self._num_images, "Too many indices"
        if self._indices is not None:
            self.record_values()
        self._indices = indices
        filepaths = self._df["filepath"].values[indices]
        for e, f in enumerate(filepaths):
            self._single_img_patches[e].load(f)
            
        if select_first:
            self.select(0)
            self._selected_image = 0
        
        
    def _update_buttons(self):
        """
        Update buttons for current selection
        """
        record = self._df.iloc[self._indices[self._selected_image]]
        self._exclude.value = bool(record["exclude"])
        for c in self._classes:
            self._selections[c].value = str(record[c])
        
    def select(self, i):
        """
        Select the ith displayed image
        """
        for e, s in enumerate(self._single_img_patches):
            if e == i:
                s.select()
            else:
                s.unselect()
        self._update_buttons()
    
    def record_values(self):
        """
        Save current button values to dataframe
        """
        i = self._indices[self._selected_image]
        self._df.loc[i, "exclude"] = self._exclude.value
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
    text = "\n### Label Counts\n *numbers show negative/positive* \n\n"
    for c in classes:
        text += "**%s:**\t%s/%s\n\n"%(c, (df[c]==0).sum(), (df[c]==1).sum())        
    return text





def pick_indices(df, pred_df, M, sort_by, subset_by):
    """
    Function to handle selecting indices of images to label
    
    :df: dataframe containing labels
    :pred_df: dataframe containing current model predictions
    :M: number of indices to retrieve
    :sort_by: method for ordering results
    :subset_by: method for subsetting df before ordering
        
    """
    # take a subset of the data to sort through
    subset = find_subset(df, subset_by)
    df = df[subset]
    pred_df = pred_df[subset]

    M = min(len(subset), M)
    # RANDOM SAMPLING
    if sort_by == "random":
        sample = df.sample(M)
    # UNCERTAINTY SAMPLING
    elif sort_by == "max entropy":
        pred_df["entropy"] = shannon_entropy(pred_df.values)
        sample = pred_df["entropy"].nlargest(M)
    # SINGLE-CLASS UNCERTAINTY SAMPLING
    elif "maxent:" in sort_by:
        col = sort_by.replace("maxent:","").strip()
        pred_df["entropy"] = shannon_entropy(pred_df[[col]].values)
        sample = pred_df["entropy"].nlargest(M)
    elif "high:" in sort_by:
        col = sort_by.replace("high: ", "")
        sample = pred_df[col].nlargest(M)
    elif "low:" in sort_by:
        col = sort_by.replace("low: ", "")
        sample = pred_df[col].nsmallest(M)
    indices = sample.index.to_numpy()
    return indices



class Labeler():
    """
    Class to manage displaying images and gathering user feedback
    """
    
    def __init__(self, classes, df, pred_df, dim=3):
        """
        :classes: list of strings; class labels
        :df: DataFrame containing filepaths and class labels
        :pred_df: dataframe of current model predictions
        :dim: dimension of the image grid to display
        :imsize: NOT YET CONNECTED
        """
        self._classes = classes
        self._dim = dim
        self._df = df
        self._pred_df = pred_df
        self._buttonpanel = ButtonPanel(classes, df, dim)
              
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
        sort_opts= ["random", "max entropy"]
        for e in ["high: ", "low: ", "maxent: "]:
            for c in self._classes:
                sort_opts.append(e+c)
            
        # generate all subsetting options
        subset_opts = ["unlabeled", "fully labeled", "partially labeled", 
                       "excluded", "not excluded"]
        for e in ["unlabeled: ", "contains: ", "doesn't contain: "]:
            for c in self._classes:
                subset_opts.append(e+c)
        
            
        self._sort_by = pn.widgets.Select(name="Sort by", options=sort_opts,
                                          value="random")
        self._retrieve_button = pn.widgets.Button(name="Sample")
        self._retrieve_button.on_click(self._retrieve_callback)
        self._subset_by = pn.widgets.Select(name="Subset by", options=subset_opts,
                                            value="unlabeled")
        
        self._update_label_button = pn.widgets.Button(name="Update Labels")
        self._label_counts = pn.pane.Markdown("")
        self._select_controls = pn.Row(
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
        indices = pick_indices(self._df, self._pred_df, self._dim**2, 
                               sort_by, subset_by)
        self._buttonpanel.load(indices)
        self._buttonpanel.label_counts.object = _generate_label_summary(self._df, self._classes)
        
 