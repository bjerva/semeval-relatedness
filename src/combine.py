#!/usr/bin/python

__author__ = 'Johannes Bjerva'
__email__  = 'j.bjerva@rug.nl'

rel_lines = sorted([line.split() for line in open('./working/foo.txt', 'r')][1:], key=lambda x:int(x[0]))
ids = set(i[0] for i in rel_lines)
rte_lines = sorted([line.split() for line in open('./working/sick.run', 'r') if line.split()[0] in ids], key=lambda x:int(x[0]))

with open('submission.txt', 'w') as out_f:
    out_f.write('pair_ID\tentailment_judgment\trelatedness_score\n')
    for i, val in enumerate(rel_lines):
        out_f.write('\t'.join(rte_lines[i][:2]) + '\t'+val[-1]+'\n')
