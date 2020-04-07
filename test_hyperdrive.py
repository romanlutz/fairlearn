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

from tempeh.execution.azureml.workspace import get_workspace
from tempeh.execution.azureml.environment_setup import configure_environment, build_package

from azureml.core import Experiment, RunConfiguration, ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig, GridParameterSampling, choice, PrimaryMetricGoal


_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier]

EXPERIMENT_NAME = 'fairlearn-hyperdrive'

def test_smoke():
    grid_size = 10
    grid_limit = 2.0

    sampling_config = GridParameterSampling(
        {
            "lambda-vec-idx": choice(list(range(grid_size))),
            "grid-size": choice(grid_size),
            "grid-limit": choice(grid_limit)
        })
    project_folder = './fairlearn-hyperdrive'
    training_script = 'run-hd.py'
    os.makedirs(project_folder, exist_ok=True)
    shutil.copy(training_script, project_folder)
    workspace = get_workspace()
    experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)
    compute_target = workspace.compute_targets['cpu-cluster']
    run_config = RunConfiguration()
    run_config.target = compute_target
    script_run_config = ScriptRunConfig(source_directory=project_folder,
                                        script=training_script,
                                        run_config=run_config)
    environment = configure_environment(
        workspace, wheel_file=build_package(package_name="fairlearn"),
        requirements_file=os.path.join("requirements.txt"))
    run_config.environment = environment
    environment.register(workspace=workspace)
    hyperdrive_config = HyperDriveConfig(run_config=script_run_config,
                                         hyperparameter_sampling=sampling_config,
                                         primary_metric_name="tradeoff_loss",
                                         primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                         max_total_runs=grid_size)
    run = experiment.submit(hyperdrive_config)
    print("submitted run")
    run.wait_for_completion(show_output=True)
    print('run completed')
    best_run = run.get_best_run_by_primary_metric()
    print(best_run.get_details()['runDefinition']['arguments'])
