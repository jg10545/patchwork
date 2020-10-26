# -*- coding: utf-8 -*-
# CODE FOR INTEGRATING WITH MLFLOW TRACKING API


def _set_up_mlflow_tracking(tracking_uri, experiment_name, run_name, notes=None,
                           param_dicts=None):
    """
    Macro to set up MLflow tracking
    
    :tracking_uri: string; Address of local or remote tracking server.
    :experiment_name: string; name of experiment to log to. if experiment doesn't exist,
        creates one with this name
    :run_name: string; name of run to log to. if run doesn't exist, one will be created
    :notes: string; any notes to add explaining the run
    :param_dicts: dictionary of dictionaries containing different parameter sets. For
        example-
        
        param_dicts={"dict1":{"a":0, "b":1}, "dict2":{"c":2}}
    
        will get turned into MLflow parameters dict1_a, dict1_b, and dict2_c
    
    Returns: a dictionary containing the MLflow client and run_id
    """
    from mlflow.tracking import MlflowClient
    # ------------------ SET UP CLIENT ------------------
    # set up an MLflow client to record stuff
    client = MlflowClient(tracking_uri)
    # ------------------ SET UP EXPERIMENT ------------------
    # does this experiment name exist yet?
    existing_experiments = [x.name for x in client.list_experiments()]
    # if it doesn't exist yet create it
    if experiment_name not in existing_experiments:
        client.create_experiment(experiment_name)
    # get the ID we'll need to reference the experiment
    expt_id = client.get_experiment_by_name(experiment_name).experiment_id
    # ------------------ SET UP RUN ------------------
    # get existing runs and their IDs
    existing_runs = {client.get_run(r.run_id).to_dictionary()["data"]["tags"]["mlflow.runName"]:r.run_id
                for r in client.list_run_infos(expt_id)}
    # does it already exist?
    if run_name in existing_runs:
        run_id = existing_runs[run_name]
    # if not make one
    else:
        tags = {}
        if run_name is not None:
            tags['mlflow.runName'] = str(run_name)
        if notes is not None:
            tags["mlflow.note.content"] = notes
        run_id = client.create_run(expt_id, tags=tags).info.run_id
        # ------------------ LOG PARAMETERS ------------------
        for d in param_dicts:
            for k in param_dicts[d]:
                client.log_param(run_id, str(d)+"_"+str(k), str(param_dicts[d][k]))
            
    return {"client":client, "run_id":run_id}
