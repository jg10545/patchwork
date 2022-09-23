import numpy as np
import pandas as pd
import holoviews as hv

hv.extension("bokeh")
from patchwork.viz._validation_scatter import _get_tsne_features, _build_scatter_holomap



def test_get_tsne_features():
    N = 31
    d = 7
    features = np.random.normal(0, 1, (N,d))
    embeds = _get_tsne_features(features)

    assert isinstance(embeds, np.ndarray)
    assert embeds.shape == (N,2)


def test_build_scatter_holomap():
    N = 25
    embeds = np.random.normal(0, 1, (N,2))
    df = pd.DataFrame({"subset":np.random.choice(["domain0", "domain1", "domain2"], size=N),
                   "filepath":["foo_%s.jpg"%i for i in range(25)]
                  })
    categories = ["cat0", "cat1", "cat2"]
    for c in categories:
        df[c] = np.random.randint(0,2,size=N)

    pred_df = pd.DataFrame({c:np.random.uniform(0, 1, size=N) for c in categories},
                      index=df.index)
    hmap = _build_scatter_holomap(df, pred_df, embeds)
    assert isinstance(hmap, hv.core.spaces.HoloMap)
