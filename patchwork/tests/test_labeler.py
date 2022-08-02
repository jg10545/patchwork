import numpy as np
import matplotlib
import panel as pn

from patchwork._labeler import _single_class_radiobuttons, _gen_figs, ButtonPanel
from patchwork._labeler import _generate_label_summary, pick_indices
from patchwork._sample import find_labeled_indices, find_excluded_indices
from patchwork._prep import prep_label_dataframe
from patchwork._badge import KPlusPlusSampler


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

    assert len(pick_indices(df, pred_df, 10, "unlabeled", "not excluded",
                            "random", "all")) == 10
    assert len(pick_indices(df, pred_df, 10, "unlabeled", "not excluded",
                            "max entropy", "all")) == 10
    assert len(pick_indices(df, pred_df, 10, "unlabeled", "not excluded",
                            "maxent: foo", "all")) == 10
    assert len(pick_indices(df, pred_df, 10, "partial", "not excluded",
                            "maxent: bar", "all")) == 10
    assert len(pick_indices(df, pred_df, 100, "unlabeled", "not excluded",
                            "maxent: foo", "all")) == 50


def test_pick_indices_with_diversity_sampling():
    # generate some fake data and labels
    N = 100
    classes = ["foo", "bar"]
    filepaths = [str(x) for x in range(N)]
    # have half labeled and half (50) unlabeled
    df = prep_label_dataframe(filepaths, classes)
    labels = np.concatenate([np.ones(int(N/2)), np.nan*np.zeros(int(N/2))])
    df = df.assign(foo=labels)
    df = df.assign(bar=labels)
    # generate predictions
    pred_df = df.copy().drop(["exclude", "filepath", "validation"], 1)
    pred_df = pred_df.assign(foo=np.linspace(0,1,N))
    pred_df = pred_df.assign(bar = np.ones(N))
    # generate features
    X = np.random.normal(0, 1, size=(N,2))
    # initialize K++ sampler- it should know that of 100 images,
    # 50 are unlabeled
    indices = list(find_labeled_indices(df)) + \
              list(find_excluded_indices(df))
    sampler = KPlusPlusSampler(X, indices=indices)
    # pull 10 of the 50 unlabeled images
    assert len(pick_indices(df, pred_df, 10, "unlabeled", "not excluded",
                            "diversity", "all", divsampler=sampler)) == 10

    # check to see that this fails politely: ask for another 100 indices;
    # there should only be 40 left to sample though.
    assert len(pick_indices(df, pred_df, 100, "unlabeled", "not excluded",
                            "diversity", "all", divsampler=sampler)) == 40









