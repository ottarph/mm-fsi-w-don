# Accompanying code for *Mesh Motion In Fluid-Structure Interaction With Deep Operator Networks*.

All experiments have been tested and run with the accompanying 
[Docker image](https://github.com/users/ottarph/packages/container/package/mm-fsi-w-don), 
which includes a build of the 2019.2.0.dev-version of ``dolfin``, necessary for the FSI-solver, 
over a cuda-compatible base image. The dataset used in the experiments is hosted 
[on zenodo](https://zenodo.org/records/12582079).

To run all experiments, launch docker container with gpu available and run the scripts ``download_data.py`` and ``run_experiments.py``.
This will produce the figures used in the manuscript, placed in ``output/figures``.
Both the hyperparameter study and the random initialization study involves a large number of runs of a lengthy training process
so running all the experiments takes a long time, approximately 29 hours on our system.

A pretrained copy of the best performing model in the hyperparameter study is included in the accompanying dataset, and is placed at 
``deeponet_extension/models/pretrained_best_run`` by running ``download_data.py``. A single model can be trained by running 
``scripts/run_deeponet_training.py`` and specifying an appropriate ``problem.json`` file, for instance the one at 
``hyperparameter_study/base_problem.json``. 

