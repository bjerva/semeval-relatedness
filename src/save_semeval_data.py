#!/usr/bin/python

"""
Data saving module for SemEval Shared Task 1.
"""

__author__ = 'Johannes Bjerva'
__email__  = 'j.bjerva@rug.nl'

import pylab as pl
import numpy as np

def write_for_evaluation(outputs, sick_ids):
    """
    Write test results to a file conforming to what is expected
    by the provided R script.
    """
    with open('./working/foo.txt', 'w') as out_f:
        out_f.write('pair_ID\tentailment_judgment\trelatedness_score\n')
        for i, line in enumerate(outputs):
            data = line

            # Fix that predictions are sometimes out of range
            if config.POST_PROCESS: 
                if data > 5.0:
                    data = 5.0
                elif data < 1.0:
                    data = 1.0


            out_f.write('{0}\t{1}\t{2}\n'.format(sick_ids[i], 'NA', data))


def plot_deviation(outputs, actual):
    pl.figure(figsize=(12, 6))
    pl.subplot(1, 2, 1)
    pl.title('Comparison')

    zipped = sorted(zip(outputs, actual), key=lambda x: x[1])
    outputs = [i[0] for i in zipped]
    actual = [i[1] for i in zipped]


    pl.plot(np.arange(len(outputs)) + 1, outputs, 'b.', 
        label='Predicted values')
    pl.plot(np.arange(len(outputs)) + 1, actual, 'r-', 
        label='Actual values')

    pl.legend(loc='upper left')
    pl.xlabel('Sentence no.')
    pl.ylabel('Relatedness')

    pl.savefig('./working/foo2.png', bbox_inches='tight')


def plot_results(regr, params, X_test, y_test, feature_names):
    """
    Plot the results from boosting iterations
    and feature evaluations, using PyLab.
    """
    ###############################################################################
    # Plot training deviance
    # Compute test set deviance
    """
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(regr.staged_decision_function(X_test)):
        test_score[i] = regr.loss_(y_test, y_pred)

    best = np.argmin(test_score)
    print "optimal", best, test_score[best]
    """

    pl.figure(figsize=(12, 10))
    pl.subplot(1, 2, 1)
    """
    pl.title('Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, regr.train_score_, 'b-', 
        label='Training Set Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', 
        label='Test Set Deviance')

    pl.legend(loc='upper right')
    pl.xlabel('Boosting Iterations')
    pl.ylabel('Deviance')
    """

    ###############################################################################
    # Plot feature importance
    feature_importance = regr.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    pl.subplot(1, 2, 2)
    pl.barh(pos, feature_importance[sorted_idx], align='center')
    pl.yticks(pos, feature_names[sorted_idx])
    pl.xlabel('Relative Importance')
    pl.title('Feature Importance')

    pl.savefig('./working/foo.png', bbox_inches='tight')

def write_to_mesh(sources, targets, ids, training):
    """
    Write features to output for use with MESH.
    Currently not used.
    """
    fname = 'rte_train.mesh' if training else 'rte_trial.mesh'
    with open(fname, 'w') as out_f:
        for i, line in enumerate(sources):
            out_f.write('Item "{0}" 1 "{1}"\n'.format(i, ids[i]))
            out_f.write('Input {0} '.format(' '.join([str(j) for j in line])))
            out_f.write('Target {0}\n\n'.format(targets[i]))
