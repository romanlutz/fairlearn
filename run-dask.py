# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fairlearn.reductions import DemographicParity
from fairlearn.reductions import GridSearch
from fairlearn.reductions._grid_search._grid_generator import _GridGenerator

from azureml.core.run import Run

from tempeh.configurations import datasets

import dask.dataframe as dd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size', type=int, help="Size of the grid")
    parser.add_argument('--grid-limit', type=float, help="Limit of the grid")

    args = parser.parse_args()
    run = Run.get_context()

    # retrieve dataset
    dataset = datasets['adult_uci']()
    X_train, X_test = dataset.get_X()
    y_train, y_test = dataset.get_y()
    A_train, A_test = X_train[:, 7], X_test[:, 7]
    
    # convert to dask dataframe
    X_train = dd.from_array(X_train).compute()
    y_train = dd.from_array(y_train).compute()
    A_train = dd.from_array(A_train).compute()

    # hack to determine lambda vector
    constraints = DemographicParity()
    constraints.load_data(X_train, y_train, sensitive_features=A_train)
    grid_search = GridSearch(SVC(), DemographicParity(),
                             grid_size=args.grid_size,
                             grid_limit=args.grid_limit)
    grid_search.fit(X_train, y_train, sensitive_features=A_train)

    # log metrics
    best_grid_index = grid_search._best_grid_index
    constraint_violation = grid_search._gammas[best_grid_index].max()
    accuracy_loss = grid_search._objectives[best_grid_index]
    tradeoff_loss = grid_search.objective_weight * accuracy_loss + \
        grid_search.constraint_weight * constraint_violation
    run.log('tradeoff_loss', tradeoff_loss)
    run.log('constraint_violation', constraint_violation)
    run.log('accuracy_loss', accuracy_loss)
    run.log('best_grid_index', best_grid_index)


if __name__ == '__main__':
    main()
