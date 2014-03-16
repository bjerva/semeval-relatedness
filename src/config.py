#!/usr/bin/python

"""
Parameters for semeval task 1
"""
import os

# Parameters
DEBUG = True            # More informative print-outs
USE_BIGRAMS = False     # Use bigrams for the DSM (Slightly worse results when this is switched on)
USE_TRIGRAMS = True     # Use trigrams for the DSM
RECALC_FEATURES = True  # Remember to switch this to True if features are changed
WRITE_TO_MESH = False    # Write to mesh (ann)
POST_PROCESS = False    # Post-process by making sure values are between 1.0 and 5.0
USE_BOXER = False        # Use boxer features
WRITE_COMPLEXITY = False # Write DRS complexity

# Paths
shared_sick = '../sick/'    # Directory containing sick files
shared_sick2 = '../sick2/'   # Directory containing alternate sick files
working_path = './working/'               # Directory containing word embeddings

# Specify word embedding file to be used
vector_num = 2
'''
if num == 1:
    vector_file = 'GoogleNews-vectors-negative300.txt'
elif num == 2:
    vector_file = 'wiki_giga_vectors.txt'
elif num == 3:
    vector_file = 'vectors_en.txt'
'''

# Temp stop list
stop_list = set(['a', 'of', 'is', 'the'])    # FIXME: Hard coded stop list

# word counts
word_freq = {}
doc_freq = {}
total_words = 0
total_sentences = 0
for id in os.listdir(shared_sick):
    path = os.path.join(shared_sick, id, 't.tok')
    for line in open(path):
        total_words += len(line.split())
        total_sentences += 1
        for word in line.split():
            word = word.lower()
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        for word in set(line.split()):
            word = word.lower()
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1
    path = os.path.join(shared_sick, id, 'h.tok')
    for line in open(path):
        total_words += len(line.split())
        total_sentences += 1
        for word in line.split():
            word = word.lower()
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        for word in set(line.split()):
            word = word.lower()
            if word in doc_freq:
                doc_freq[word] += 1
            else:
                doc_freq[word] = 1