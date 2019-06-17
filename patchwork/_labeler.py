"""

                _modelpicker.py


GUI code for training a model

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
from PIL import Image


from patchwork._sample import find_subset
from patchwork._util import shannon_entropy

def _reformat(x):
    return str(x).split("\nName")[0].replace("\n", "<br>")

def _generate_label_summary(df, classes):
    text = ""
    for c in classes:
        text += "**%s:**\t%s/%s\n\n"%(c, (df[c]==0).sum(), (df[c]==1).sum())        
    return text

#def single_image_chooser(classes, imsize=100):
#    img = pn.panel("default_img.gif", width=imsize, height=imsize)
#    class_choice = pn.widgets.RadioBoxGroup(name="Class", options=classes)
#    return pn.Row(img, class_choice)

def _load_to_fig(f):
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(Image.open(f))
    ax.axis("off")
    plt.close(fig) # found in github issue, not sure it'll work
    return fig

def _single_class_radiobuttons(width=125, height=25):
    return pn.widgets.RadioButtonGroup(options=["None", "0", "1"], width=width, align="center", 
                                       height=height, value="None")



class SinglePatchLabeler(object):
    """
    panel widget that has all the pieces for labeling a single patch
    """
    def __init__(self, classes):#, size=100):
        self._index = None
        self._filepath = None
        self._classes = classes
        self._value_map = {"None":None, "0":0, "1":1}
        self._selections = {c:_single_class_radiobuttons() for c in classes}
        self._exclude = pn.widgets.Checkbox(name="exclude", align="center")
        
        button_panel = pn.Column(self._exclude, 
                                 *[
                                     pn.Row(self._selections[c], 
                                            pn.pane.Markdown(c, align="center"), 
                                            height=25)
                                     for c in classes], width=200) 

        
        fig, ax = plt.subplots()
        ax.plot([])
        ax.axis("off")
        self._fig_panel = pn.pane.Matplotlib(fig, width=150, height=100, margin=0)#, width=size, height=size)
        self.panel = pn.Row(self._fig_panel, button_panel,
                            width_policy="fixed")
        
        
    def update(self, index, record):
        """
        Update the widget for a new record
        
        :index: dataframe index for the record (so we know what to write back to later)
        :record: row from dataframe, containing filepath, exclude, and class columns
        """
        self._index = index
        self._filepath = record["filepath"]
        
        #plt.close(self._fig_panel.object)
        self._fig_panel.object = _load_to_fig(self._filepath)
        self._exclude.value = bool(record["exclude"])
        
        for c in self._classes:
            self._selections[c].value = str(record[c])
        
        
        
    def get(self):
        """
        Query the current state of the widget
        
        Returns the row index for the dataframe and a dictionary
        mapping columns to values
        """
        outdict = {"exclude":self._exclude.value}
        for c in self._classes:
            outdict[c] = self._value_map[self._selections[c].value]
            
        return self._index, outdict
    
    def __call__(self, df):
        """
        Updates a dataframe with the current state of the widget
        """
        i, valdict = self.get()
        for c in valdict:
            df.loc[i, c] = valdict[c]
            
        return df


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
    #if labeled:
    #    subset = df[pd.notnull(df["label"])]
    #else:
    #    subset = df[pd.isnull(df["label"])]
            
    #if len(subset) > 0:
    M = min(len(subset), M)
    # RANDOM SAMPLING
    if sort_by == "random":
        sample = df.sample(M)
        #indices = sample.index.to_numpy()
    # UNCERTAINTY SAMPLING
    elif sort_by == "max entropy":
        pred_df["entropy"] = shannon_entropy(pred_df.values)
        sample = pred_df["entropy"].nlargest(M)
        #indices = sample.index.to_numpy()
    # SINGLE-CLASS UNCERTAINTY SAMPLING
    elif "maxent:" in sort_by:
        col = sort_by.replace("maxent:","").strip()
        pred_df["entropy"] = shannon_entropy(pred_df[[col]].values)
        sample = pred_df["entropy"].nlargest(M)
        #indices = sample.index.to_numpy()
    elif "high:" in sort_by:
        col = sort_by.replace("high: ", "")
        #if col in df.columns:
        sample = pred_df[col].nlargest(M)
        #indices = sample.index.to_numpy()
    elif "low:" in sort_by:
        col = sort_by.replace("low: ", "")
        #if col in df.columns:
        sample = pred_df[col].nsmallest(M)
    indices = sample.index.to_numpy()
    return indices



class Labeler():
    """
    Class to manage displaying images and gathering user feedback
    """
    
    def __init__(self, classes, df, pred_df, dim=2, imsize=100):
        """
        
        """
        self._classes = [x for x in df.columns if x not in ["filepath", "exclude"]]
        #self._classes = classes
        #self._label_types = [None, "(exclude)"] + list(classes)
        self._dim = dim
        self._df = df
        self._pred_df = pred_df
        self._indices = "nuthin here"
        
        #self._choosers = [single_image_chooser(self._label_types, imsize)
        #                  for _ in range(dim**2)]
        # create an object for every patch labeler we'll display simultaneously
        self._choosers = [SinglePatchLabeler(self._classes)
                            for _ in range(dim**2)]
        # now lay them out on a grid
        # sizing mode had been "stretch_both"
        self.GridSpec = pn.GridSpec(sizing_mode="stretch_both")#"fixed")#,
                                    #width=800, height=600)
                                 #max_width=50)
        i = 0
        for r in range(dim):
            for c in range(dim):
                self.GridSpec[r,c] = self._choosers[i].panel
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
        self._retrieve_watcher = self._retrieve_button.param.watch(
                        self._retrieve_callback, ["clicks"])
        #self._whether_labeled_radio = pn.widgets.RadioBoxGroup(name="labeled", 
        #                                        options=["unlabeled", "labeled"])
        self._subset_by = pn.widgets.Select(name="Subset by", options=subset_opts,
                                            value="unlabeled")
        
        self._update_label_button = pn.widgets.Button(name="Update Labels")
        self._update_watcher = self._update_label_button.param.watch(
                        self._update_label_callback, ["clicks"])
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
        #labeled = self._whether_labeled_radio.value == "labeled"
        indices = pick_indices(self._df, self._pred_df, self._dim**2, 
                               sort_by, subset_by)
        #indices = pick_indices(self._df, self._dim**2, 
        #                       sort_by, labeled)
        if len(indices) > 0:
            #self._fill_images(indices)
            for j,i in enumerate(indices):
                self._choosers[j].update(i, self._df.iloc[i])
            
    def _update_label_callback(self, *events):
        for c in self._choosers:
            c(self._df)
        #labels = self._df["label"].values
        #for i, ind in enumerate(self._indices):
        #    newval = self._choosers[i][1].value
        #    labels[ind] = newval
        #self._df["label"] = labels
        
        #self._label_counts.object = _reformat(self._df["label"].value_counts())
        self._label_counts.object = _generate_label_summary(self._df, self._classes)