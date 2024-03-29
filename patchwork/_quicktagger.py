# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import panel as pn

from patchwork._sample import PROTECTED_COLUMN_NAMES, find_partially_labeled


class SingleImageTagger():
    
    def __init__(self, f, classname="class", size=200):
        self.classname = classname
        # determine PNG or JPG and build image pane
        if f.lower().endswith(".png"):
            self._pane = pn.pane.PNG(f, width=size, height=size, sizing_mode="fixed")
        elif f.lower().endswith(".jpg") or f.lower().endswith(".jpeg"):
            self._pane = pn.pane.JPG(f, width=size, height=size, sizing_mode="fixed")
        else:
            assert False, "can't determine file format"
        # build button
        self._button = pn.widgets.Button(name=f"doesn't contain {classname}", width=size)
        
        # build column to hold it
        self.panel = pn.Column(self._pane, self._button, background="white")
        
        # attach callback to button
        def _callback(*events):
            if self.panel.background == "white":
                self.panel.background = "blue"
                self._button.name = f"contains {self.classname}"
            elif self.panel.background == "blue":
                self.panel.background = "red"
                self._button.name = "EXCLUDE"
            else:
                self.panel.background = "white"
                self._button.name = f"doesn't contain {self.classname}"
                
        self._button.on_click(_callback)
    
    def __call__(self):
        # return label: 0, 1, or np.nan
        _parse_class = {"white":0, "blue":1, "red":np.nan}
        return _parse_class[self.panel.background]
    
    def update_image(self, f, classname=None):
        if classname is not None:
            self.classname = classname
        # update image pane to new image, and reset button
        self._pane.object = f
        self.panel.background = "white"
        self._button.name = f"doesn't contain {self.classname}"





class QuickTagger():
    """
    Barebones panel app for quickly tagging multiclass images. Call QuickTagger.panel
    in a notebook to get to the GUI.
    
    Takes inputs in the same form as the active learning tool- use pw.prep_label_dataframe()
    to get started.
    """
    def __init__(self, df, outfile=None, size=200):
        """
        :df: pandas DataFrame containing filepaths to images and a label column
            for each category
        :outfile: optional; path to CSV to write to during tagging
        :size: scaling number for panel GUI
        """
        self.df = df.copy()
        self.outfile = outfile
        self.categories = [c for c in df.columns if c not 
                           in PROTECTED_COLUMN_NAMES]
        
        self._set_up_panel(size)
        if "subset" in df.columns:
            self._stratify_categories = df.subset.unique()
        
    def _set_up_panel(self, size=200):
        init_sample_indices = np.random.choice(np.arange(len(self.df))[pd.isna(self.df[self.categories[0]])], 9)
        self._current_indices = init_sample_indices
        self._taggers = [SingleImageTagger(self.df.loc[i,"filepath"], self.categories[0], size) 
                         for i in init_sample_indices]
        self._grid = pn.GridBox(*[t.panel for t in self._taggers], ncols=3, nrows=3, width=3*size)
        self._classchooser = pn.widgets.Select(options=self.categories, 
                                               value=self.categories[0], name="Class to annotate", width=size)
        self._strategy = pn.widgets.Select(options=["uniform", "stratified"],
                                           name="Sampling strategy", width=size)
        
        self._samplesavebutton = pn.widgets.Button(name="save and sample", button_type="primary", width=size)
        self._samplebutton = pn.widgets.Button(name="sample without saving", button_type="danger", width=size)
        self._backbutton = pn.widgets.Button(name="go back", button_type="warning", width=size)
        self._summaries = pn.pane.DataFrame(self._compute_summaries(), index=False, width=size)
        self.panel = pn.Row(
            self._grid,
            pn.Spacer(width=50),
            pn.Column(
                self._classchooser,
                self._strategy,
                self._samplesavebutton,
                self._samplebutton,
                self._backbutton,
                self._summaries
            )
        )
        # set up button callbacks
        self._samplebutton.on_click(self._sample)
        self._samplesavebutton.on_click(self._record_and_sample_callback)
        self._backbutton.on_click(self._go_back_callback)
        
        
    def serve(self, **kwargs):
        """
        wrapper for panel.serve()
        """
        p = self.panel
        pn.serve(p, title="quicktagger", **kwargs)
        
    def _get_uniform_sampled_indices(self):
        cat = self._classchooser.value
        available_indices = np.arange(len(self.df))[pd.isna(self.df[cat])]
        return np.random.choice(available_indices, size=9, replace=False)
    
    def _get_stratified_sampled_indices(self):
        assert hasattr(self, "_stratify_categories"), "need a subset column for stratified sampling"
        cat = self._classchooser.value
        indices = np.arange(len(self.df))
        unlab = pd.isna(self.df[cat])
        
        sampled = []
        while len(sampled) < 9:
            # pick a subset
            subset = np.random.choice(self._stratify_categories)
            available_indices = indices[unlab&(self.df.subset == subset)]
            if len(available_indices) > 0:
                sampled.append(np.random.choice(available_indices))
        return np.array(sampled)
        
        
        
    def _sample(self, *events):
        cat = self._classchooser.value
        self._previous_indices = self._current_indices.copy()
        if self._strategy.value == "uniform":
            self._current_indices = self._get_uniform_sampled_indices()
        else:
            self._current_indices = self._get_stratified_sampled_indices()
            
        for e,i in enumerate(self._current_indices):
            self._taggers[e].update_image(self.df.filepath.values[i], cat)
            
    def _go_back_callback(self, *events):
        self._current_indices = self._previous_indices
        for e,i in enumerate(self._current_indices):
            self._taggers[e].update_image(self.df.filepath.values[i],
                                          self._classchooser.value)
        
    def _record(self):
        for i, t in zip(self._current_indices, self._taggers):
            self.df.loc[i, t.classname] = t()
            
    def _record_and_sample_callback(self, *events):
        self._record()
        self._sample()
        self.save()
        self._summaries.object = self._compute_summaries()

    def _compute_summaries(self):
        cats = self.categories
        label_counts = []
        for c in cats:
            summary = {"class":c, "None":np.sum(pd.isna(self.df[c])), 
                       "0":np.sum(self.df[c] == 0), "1":np.sum(self.df[c] == 1)}
            label_counts.append(summary)
            
        return pd.DataFrame(label_counts, columns=["class", "None", "0", "1"])
    
    def save(self):
        """
        Save only the parts of the dataframe that have at least one label
        """
        if self.outfile is not None:
            keep = np.array([False]*len(self.df))
            for c in self.categories:
                keep = keep|pd.notna(self.df[c]).values
            
            self.df[keep].to_csv(self.outfile, index=False)
            
    def save_all(self, outfile=None):
        """
        Save the entire dataframe, including unlabeled rows
        """
        if outfile is None:
            if self.outfile is not None:
                outfile = self.outfile
            else:
                assert False, "i don't know where to save this"
                
        self.df.to_csv(outfile, index=False)
    






