#!/usr/bin/python
# -*- coding: utf-8 -*-


import os
import string
import itertools

shared_sick = '/home/rob/candc/working/sick/'
shared_sick2 = '/home/rob/candc/working/sick2/'
instances = os.listdir(shared_sick)

writeFile = open('test.out', 'w')
negations = {'not':'', 'n\'t':'', 'no':'a', 'none':'some', 'nobody':'somebody'}


if not os.path.isdir(shared_sick2):
 os.makedirs(shared_sick2)
 
 

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


paraphrases = {}
for line in open('/home/rob/candc/working/ppdb.1'):
    source = line.split('|')[1][1:-1]
    target = line.split('|')[4][1:-1]
    if source in paraphrases:
        paraphrases[source].append(target)
    else:
        paraphrases[source] = [target]

def getReplacements(t, h):
    replacements = []
    for wordT in t:
         if wordT in paraphrases:
             for replacementWord in paraphrases[wordT]:
                 if replacementWord in h:
                     replacement = [t.index(wordT), replacementWord, wordT]
                     if replacement in replacements:
                         pass #hoe kan dit voorkomen?
                     else:
                         replacements.append(replacement)
                    
    for wordT in t:
         if wordT in negations:
             replacements.append([t.index(wordT), negations[wordT], wordT])
    for wordH in h:
         if wordH in negations:
             replacements.append([len(t)+h.index(wordH), negations[wordH], wordH])
    return replacements
    
for id in instances:
    sentence_a = open(os.path.join(shared_sick, id, 't.tok'), 'r').read().split()
    sentence_b = open(os.path.join(shared_sick, id, 'h.tok'), 'r').read().split()
    sentence_a[0] = sentence_a[0].lower()
    sentence_b[0] = sentence_b[0].lower()
    replacements = getReplacements(sentence_a, sentence_b)
    combination_counter = 1
    for combination in list(powerset(replacements)):
        folder = os.path.join(shared_sick2,'{0}.{1}'.format(id, combination_counter))
        if combination == ():
            pass
        else:
            os.makedirs(folder)
            write_t = open(os.path.join(folder, 't'),'w')
            write_h = open(os.path.join(folder, 'h'),'w')
            new_t = sentence_a
            new_h = sentence_b
            for replacement in combination:
                if replacement[0] < len(sentence_a):
                    new_t[replacement[0]] = replacement[1]
                else:
                    new_h[replacement[0]-len(sentence_a)] = replacement[1]
                
            write_t.write(string.join(new_t,' '))
            write_t.write('\n')
            write_h.write(string.join(new_h,' '))
            write_h.write('\n')
            for replacement in combination:
                if replacement[0] < len(sentence_a):
                    new_t[replacement[0]] = replacement[2]
                else:
                    new_h[replacement[0]-len(sentence_a)] = replacement[2]
            
            combination_counter += 1