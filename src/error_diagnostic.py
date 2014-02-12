#!/usr/bin/python

"""
Error diagnostic module for SemEval Shared Task 1.
"""

__author__ = 'Johannes Bjerva'
__email__  = 'j.bjerva@rug.nl'

def output_errors(outputs, gold, sick_ids, sick_sentences):
    """
    For each item with an absolute error > 1.0,
    print the item to an error file for further analysis.
    """
    with open('./working/err.txt', 'w') as out_f:
        out_f.write('pair_ID\tdiff\tpred\tcorr\tsentence1\tsentence2\n')
        errs = []
        for i, line in enumerate(outputs):
            data = line
            corr = gold[i]
            diff = abs(data-corr)
            if diff > 1.0:
                errs.append((sick_ids[i], round(diff, 1), round(data, 1), corr, ' '.join(sick_sentences[i][0]), ' '.join(sick_sentences[i][1])))

        errs.sort(key=lambda x:-x[1])

        for line in errs:
            out_f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(*line))
