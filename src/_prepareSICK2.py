#!/usr/bin/python


import os
import string

shared_sick = 'working/sick/'
shared_sick2 = 'working/sick2/'
instances = os.listdir(shared_sick)

if not os.path.isdir(shared_sick2):
 os.makedirs(shared_sick2)

paraphrases = {}
for line in open('working/ppdb.1'):
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
                 if replacementWord in h and replacementWord not in t and wordT not in h:
                     replacements.append([wordT, replacementWord])
    return replacements


for id in instances:
    sentence_a = open(os.path.join(shared_sick, id, 't.tok'), 'r').read().split()
    sentence_b = open(os.path.join(shared_sick, id, 'h.tok'), 'r').read().split()
    replacement_counter = 1
    for replacement in getReplacements(sentence_a, sentence_b):
        idFolder = os.path.join(shared_sick2, '{0}.{1}'.format(str(id),replacement_counter))
        os.mkdir(idFolder)
        sentence_a[sentence_a.index(replacement[0])] = replacement[1]
        write_t = open(os.path.join(idFolder, 't'), 'w')
        write_h = open(os.path.join(idFolder, 'h'), 'w')
        sentence_a.append('\n')
        sentence_b.append('\n')
        write_t.write(string.join(sentence_a, ' '))
        write_h.write(string.join(sentence_b, ' '))
        sentence_a[sentence_a.index(replacement[1])] = replacement[0]
        replacement_counter += 1