# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
import platform
import pytest
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, \
    EqualizedOdds

from tempeh.execution.azureml.workspace import get_workspace
from tempeh.execution.azureml.environment_setup import configure_environment, build_package

from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, ComputeTarget
from azureml.core.runconfig import MpiConfiguration
from azureml.core.compute import AmlCompute
from azureml.train.estimator import Estimator

import dask
from dask.distributed import Client
#from dask_cloudprovider import AzureMLCluster


_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier]

EXPERIMENT_NAME = 'fairlearn-dask'
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP_NAME")
COMPUTE_NAME = 'dask-cluster'
NODE_COUNT = 3


def test_smoke():
    grid_size = 10
    grid_limit = 2.0

    project_folder = './fairlearn-dask'
    training_script = 'run-dask.py'
    os.makedirs(project_folder, exist_ok=True)
    shutil.copy(training_script, project_folder)

    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_DS13_V2',
        min_nodes=0,
        max_nodes=NODE_COUNT,
        #vnet_name="vnet",
        #subnet_name="subnet",
        #vnet_resourcegroup_name=RESOURCE_GROUP,
        idle_seconds_before_scaledown=300
    )
    workspace = get_workspace(compute_name=COMPUTE_NAME, compute_config=compute_config)
    experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)
    run_config = RunConfiguration()
    run_config.target = workspace.compute_targets[COMPUTE_NAME]
    run_config.mpi = MpiConfiguration()
    run_config.node_count = NODE_COUNT
    script_run_config = ScriptRunConfig(source_directory=project_folder,
                                        script=training_script,
                                        run_config=run_config,
                                        arguments=['--grid-limit', grid_limit,
                                                   '--grid-size', grid_size])
    environment = configure_environment(
        workspace, wheel_file=build_package(package_name="fairlearn"),
        requirements_file=os.path.join("requirements.txt"))
    environment.python.conda_dependencies.add_pip_package('dask[complete]')
    environment.python.conda_dependencies.add_pip_package('dask-ml[complete]')
    run_config.environment = environment
    environment.register(workspace=workspace)

    #cluster = AzureMLCluster(workspace, compute_target, environment)

    run = experiment.submit(script_run_config)
    print("submitted run")
    run.wait_for_completion(show_output=True)
    print('run completed')
    print("best grid index: {}".format(run.get_metrics("best_grid_index")))
