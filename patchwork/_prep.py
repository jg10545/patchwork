"""

            _prep.py


"""

import pandas as pd


def prep_label_dataframe(filepaths, classes):
    """
    Build a pandas DataFrame for labeling
    
    :filepaths: list of strings; paths to each image file
    :classes: list of strings; name of each category to label
    """
    df = pd.DataFrame({"filepath":filepaths})
    df["exclude"] = False
    df["validation"] = False
    for c in classes:
        df[c] = None
    return df