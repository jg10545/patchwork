import numpy as np
import matplotlib
import panel as pn

from patchwork._labeler import _single_class_radiobuttons, _gen_figs, ButtonPanel
from patchwork._labeler import _generate_label_summary, pick_indices
from patchwork._prep import prep_label_dataframe


def test_gen_figs():
    testfigs = _gen_figs([np.random.uniform(0,1,(256,256)) for _ in range(9)])
    assert isinstance(testfigs[0], matplotlib.figure.Figure)
    assert len(testfigs) == 9
    
    
def _test_single_class_radiobuttons():
    buttons = _single_class_radiobuttons()
    assert isinstance(buttons, pn.widgets.select.RadioButtonGroup)




def test_buttonpanel():
    def _mock_load_func(x):
        """
        randomly generate array instead of loading an image file
        from disk to an array
        """
        return np.random.uniform(0, 1, (256,256, 3))
    
    classes = ["foo", "bar"]
    filepaths = ["foo.jpg", "bar.jpg"]
    df = prep_label_dataframe(filepaths, classes)
    
    bp = ButtonPanel(classes, df, _mock_load_func)
    bp.load(np.array([0]))
    bp.record_values()
    
    
    
    
def test_pick_indices():
    N = 100
    classes = ["foo", "bar"]
    filepaths = [str(x) for x in range(N)]
    df = prep_label_dataframe(filepaths, classes)
    foo = np.concatenate([np.ones(int(N/2)), np.nan*np.zeros(int(N/2))])
    df = df.assign(foo=foo)
    
    pred_df = df.copy().drop(["exclude", "filepath", "validation"], 1)
    pred_df = pred_df.assign(foo=np.linspace(0,1,N))
    pred_df = pred_df.assign(bar = np.ones(N))
    
    assert len(pick_indices(df, pred_df, 10, "random", "unlabeled")) == 10
    assert len(pick_indices(df, pred_df, 10, "max entropy", "unlabeled")) == 10
    assert len(pick_indices(df, pred_df, 10, "maxent: foo", "unlabeled")) == 10
    assert len(pick_indices(df, pred_df, 10, 
                            "maxent: bar", "partially labeled")) == 10
    assert len(pick_indices(df, pred_df, 100, "maxent: foo", "unlabeled")) == 50
    
    
    
    
    
    
    
    
    
    
    