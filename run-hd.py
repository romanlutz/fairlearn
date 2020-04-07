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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda-vec-idx', type=int,
                        help='Index of the lambda vector in the grid')
    parser.add_argument('--grid-size', type=int, help="Size of the grid")
    parser.add_argument('--grid-limit', type=float, help="Limit of the grid")

    args = parser.parse_args()
    run = Run.get_context()
    run.log('lambda_vec_idx', args.lambda_vec_idx)

    # retrieve dataset
    dataset = datasets['adult_uci']()
    X_train, X_test = dataset.get_X()
    y_train, y_test = dataset.get_y()
    A_train, A_test = X_train[:, 7], X_test[:, 7]

    # hack to determine lambda vector
    constraints = DemographicParity()
    constraints.load_data(X_train, y_train, sensitive_features=A_train)
    grid = _GridGenerator(grid_size=args.grid_size,
                          grid_limit=args.grid_limit,
                          pos_basis=constraints.pos_basis,
                          neg_basis=constraints.neg_basis,
                          neg_allowed=constraints.neg_basis_present,
                          force_L1_norm=False).grid
    lambda_vec = grid[args.lambda_vec_idx]
    single_entry_grid = pd.DataFrame(lambda_vec)
    # rename index in the grid to 0 so that Grid Search doesn't get confused.
    single_entry_grid = single_entry_grid.rename(columns={args.lambda_vec_idx: 0})

    # apply mitigation according to lambda vector
    grid_point_eval = GridSearch(SVC(), DemographicParity(),
                                 grid_size=args.grid_size,
                                 grid_limit=args.grid_limit,
                                 grid=single_entry_grid)
    grid_point_eval.fit(X_train, y_train, sensitive_features=A_train)

    # log metrics
    constraint_violation = grid_point_eval._gammas[0].max()
    accuracy_loss = grid_point_eval._objectives[0]
    tradeoff_loss = grid_point_eval.objective_weight * accuracy_loss + \
        grid_point_eval.constraint_weight * constraint_violation
    run.log('tradeoff_loss', tradeoff_loss)
    run.log('constraint_violation', constraint_violation)
    run.log('accuracy_loss', accuracy_loss)


if __name__ == '__main__':
    main()
