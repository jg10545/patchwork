import numpy as np
import bokeh.models.tools
import sklearn.manifold, sklearn.preprocessing
import warnings

try:
    import holoviews as hv
except:
    warnings.warn("holoviews not installed")

from patchwork._sample import PROTECTED_COLUMN_NAMES


def _get_tsne_features(features):
    """
    Normalize and compute T-SNE embeddings for an (N,d) array of features
    """
    features = sklearn.preprocessing.Normalizer().fit_transform(features.astype(np.float64))
    return sklearn.manifold.TSNE(2, init="pca").fit_transform(features)


def _scatter(d, p, c):

    d = d.copy()
    d["entropy"] = -1*d[c].values*np.log2(p[c].values+1e-5) - \
                        (1-d[c].values)*np.log2(1-p[c].values+1e-5)
    d["pred"] = p[c].apply(lambda x: int(x >= 0.5))
    TOOLTIPS = """
        <img src="@filepath" height="128" alt="@imgs" width="128"></img>
    """
    hovertool = bokeh.models.tools.HoverTool(tooltips=TOOLTIPS)

    points = hv.Points(d, kdims=["x", "y"]).opts(color=hv.dim("subset"),
                                                 size=4*(1+hv.dim("entropy")),
                               alpha=0.5, cmap="glasbey_dark", marker="o", 
                               tools=[hovertool],
                               width=600, height=500, xaxis=None, yaxis=None)

    pos = hv.Points(d[d[c] == 1], kdims=["x", "y"]).opts(size=4*(1+hv.dim("entropy")),
                                                         alpha=1, line_width=2,
                                        marker="o", fill_color=None, line_color="darkblue")
    error = hv.Points(d[d[c] != d.pred], kdims=["x", "y"]).opts(size=4*(1+hv.dim("entropy")), alpha=1,
                                              line_width=2, marker="x", fill_color=None,
                                              line_color="firebrick")
    return points*pos*error


def _build_scatter_holomap(df, pred_df, embeds, return_dict=False):
    """
    :df: dataframe of filepaths, labels, and subsets
    :pred_df: dataframe of predictions
    :embeds: (N,2) array of low-dimensional embeddings (T-SNE or other)
    """
    categories = [c for c in df.columns if c not in PROTECTED_COLUMN_NAMES]
    df = df.copy()
    df["x"] = embeds[:,0]
    df["y"] = embeds[:,1]
    
    if return_dict:
        return {c:_scatter(df, pred_df, c) for c in categories}
    else:
        return hv.HoloMap({c:_scatter(df, pred_df, c) for c in categories})





