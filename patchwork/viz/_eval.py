import holoviews as hv




def error_vs_n_plot(results, testerr=True, width=900, height=600, logx=True, cmap="category10",
                    alpha=0.75, size=10, **kwargs):
    """
    Visualize results of patchwork.sample_and_evaluate()
    """
    vdims = [c for c in results.columns if c not in ["N", "train_error", "test_error"]]
    if testerr:
        y = "test error"
    else:
        y = "train error"
    fig = hv.Scatter(results, kdims=["N", y], vdims=vdims).opts(color="fcn", tools=["hover"], size=size,
                                                             width=width, height=height, logx=logx,
                                                                cmap=cmap, show_grid=True, alpha=alpha,
                                                               **kwargs)
    return fig
