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
    with open('working/foo.txt', 'w') as out_f:
        out_f.write('pair_ID\tentailment_judgment\trelatedness_score\n')
        for i, line in enumerate(outputs):
            out_f.write('{0}\t{1}\t{2}\n'.format(sick_ids[i], 'NA', line))

def plot_results(regr, params, X_test, y_test, feature_names):
    """
    Plot the results from boosting iterations
    and feature evaluations, using PyLab.
    """
    ###############################################################################
    # Plot training deviance
    # Compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(regr.staged_decision_function(X_test)):
        test_score[i] = regr.loss_(y_test, y_pred)

    pl.figure(figsize=(12, 6))
    pl.subplot(1, 2, 1)
    pl.title('Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, regr.train_score_, 'b-', 
        label='Training Set Deviance')
    pl.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', 
        label='Test Set Deviance')
    pl.legend(loc='upper right')
    pl.xlabel('Boosting Iterations')
    pl.ylabel('Deviance')

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

    pl.savefig('working/foo.png', bbox_inches='tight')

def write_to_mesh(data):
    """
    Write features to output for use with MESH.
    Currently not used.
    """
    with open('rte.mesh', 'w') as out_f:
        c = 0
        for line in data:
            rep = sentence_composition(line[1], line[2])
            out = line[4]
            out_f.write('Item "{0}" 1 "null"\n'.format(c))
            out_f.write('Input {0} '.format(' '.join([str(i) for i in rep])))
            out_f.write('Target {0}\n\n'.format(' '.join(['1' 
                        if i == out else '0' for i in xrange(3)])))
            c += 1