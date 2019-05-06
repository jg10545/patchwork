import numpy as np
import matplotlib.pyplot as plt
import os
import yaml



def run_experiment(savepath, classifier, num_experiments, feature_vecs, imfiles, num_calls,
                   groundtruth, testx, testy, **kwargs):
    """
    
    """
    outdict = {"name":savepath.split("/")[-1].split(".")[0], "test_auc":[]}
    
    for n in range(num_experiments):
        print("Running experiment %s of %s"%(n,num_experiments))
        obj = classifier(feature_vecs, imfiles, **kwargs)
        obj(num_calls=num_calls, groundtruth=groundtruth, testx=testx, testy=testy)
    
        outdict["epsilon"] = obj._epsilon
        outdict["epochs"] = obj._epochs
        outdict["test_auc"].append([float(x) for x in obj.test_auc])
        
        yaml.dump(outdict,
                  open(savepath, "w"),
                  default_flow_style=False)
        
        
def show_experiments(expdir="experiments/"):
    colors = plt.cm.get_cmap("Dark2").colors
    for i, f in enumerate(os.listdir(expdir)):
        c = colors[i%8]
        loaded = yaml.load(open(os.path.join(expdir,f)))
        for j, t in enumerate(loaded["test_auc"]):
            if j == 0:
                plt.plot(t, color=c, label=loaded["name"], alpha=0.5)
            else:
                plt.plot(t, color=c, alpha=0.5)
    plt.legend(loc="lower right")
    plt.xlabel("labeling step", fontsize=14)
    plt.ylabel("test AUC", fontsize=14)
    plt.grid(True)