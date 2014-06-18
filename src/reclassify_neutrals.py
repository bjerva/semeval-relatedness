#!/usr/bin/python

__author__ = 'Johannes Bjerva'
__email__  = 'j.bjerva@rug.nl'

from sklearn import svm
from collections import defaultdict
import config

def read_gold():
    # Read all RTE data
    with open(config.working_path+'SICK_all.txt', 'r') as in_f:
        gold_rte = dict([[line.strip().split('\t')[0], line.strip().split('\t')[-1]] for line in in_f][1:]) #ID, RTE
        print len(gold_rte)

    # Read Johan's predictions, when 'NEUTRAL'
    with open(config.working_path+'sick.run', 'r') as in_f:
        estimated_rte = dict(line.strip().split('\t')[0:2] for line in in_f if line.strip().split('\t')[1] == 'NEUTRAL') #ID, RTE
        print 'Estimated as NEUTRAL:', len(estimated_rte)

    # Read gold relatedness data (redundantly coded)
    with open(config.working_path+'SICK_all.txt', 'r') as in_f:
        estimated_relatedness = dict([[line.strip().split('\t')[0], line.strip().split('\t')[-2]] for line in in_f][1:]) #ID, relatedness
    
    # Read our system's predictions
    wait = set()
    with open(config.working_path+'foo.txt', 'r') as in_f:
        for i in [[line.strip().split('\t')[0], line.strip().split('\t')[-1]] for line in in_f][1:]:
            if i[0] in estimated_rte:
                wait.add(i[0])
                estimated_relatedness[i[0]] = i[1] #ID, relatedness
        print len(estimated_relatedness), len(wait)

    # Make sure test data is last in our lists
    all = [i for i in xrange(10001) if i not in wait]
    wait = list(wait)

    # Reordering data sets
    training = [float(estimated_relatedness[str(i)]) for i in all+wait if str(i) in estimated_rte]
    targets = [rte_dict[gold_rte[str(i)]] for i in all+wait if str(i) in estimated_rte]
    old = [rte_dict[estimated_rte[str(i)]] for i in all+wait if str(i) in estimated_rte]
    
    # Just to be safe
    assert len(training) == len(targets) == len(old)

    return training, targets, old, wait

def train(X, y, old, trial_order):

    # Split where test data ends
    split = len(X)-len(trial_order)
    X = [[i] for i in X]

    # Simple classifier
    clf = svm.SVC()
    clf.fit(X[:split], y[:split])

    corrected, unchanged, error = 0, 0, 0
    out_f = open('sick_corr.run', 'w')
    for i, val in enumerate(X[split:]):
        # Write new prediction to file
        out_f.write('{0} {1} {2}\n'.format(trial_order[i], reverse[clf.predict(val)[0]], str(val[0])))

        # Naive evaluation
        if old[split+i] != y[split+i]:
            if clf.predict(val) == y[split+i]:
                corrected += 1
            else:
                unchanged += 1
        else:
            if clf.predict(val) == y[split+i]:
                unchanged += 1
            else:
                error += 1

    # Print evaluation (only possible for trial data)
    print corrected, error, unchanged
    print corrected / float(error+corrected)

rte_dict = {'NEUTRAL':0, 'CONTRADICTION':2, 'ENTAILMENT':1}
reverse = {0:'NEUTRAL',1:'ENTAILMENT',2:'CONTRADICTION'}
if __name__ == '__main__':
    sources, targets, old, trial_order = read_gold()
    train(sources, targets, old, trial_order)

