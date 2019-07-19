# -*- coding: utf-8 -*-
import pandas as pd
from patchwork._prep import prep_label_dataframe


def test_prep_label_dataframe():
    filepaths = ["foo.png", "bar.png", "foobar.png"]
    classes = ["shoe", "biscotti", "punctuality"]
    
    df = prep_label_dataframe(filepaths, classes)
    assert len(df) == 3
    assert "exclude" in df.columns
    assert "shoe" in df.columns
    assert (df["exclude"] == False).sum() == 3
    assert (pd.isnull(df["punctuality"])).sum() == 3