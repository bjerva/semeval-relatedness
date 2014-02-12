#!/usr/bin/python

"""
Feature extraction module for SemEval Shared Task 1.
"""

__author__ = 'Johannes Bjerva, and Rob van der Goot'
__email__  = 'j.bjerva@rug.nl'

import os
import requests
import numpy as np

from collections import defaultdict
from scipy.spatial.distance import cosine
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

import drs_complexity
import load_semeval_data
import config

def get_replacements(t, h):
    """
    Return all posible replacements for sentence t to become more like sentence h.
    """
    replacements = []
    for wordT in t:
         if wordT in paraphrases:
             for replacementWord in paraphrases[wordT]:
                 if replacementWord in h and replacementWord not in t and wordT not in h:
                     replacements.append([wordT, replacementWord])
    return replacements

def get_number_of_instances(modelfile):
    """
    Return the number of instances in the modelfile.
    """
    firstLine = open(modelfile, 'r').readline()
    return float(len(firstLine.split('d'))-2)
  
def instance_overlap(id, sentence_a, sentence_b):
    """
    Calculate the amount of overlap between the instances in the models of sentence_a (t) and sentence_b (h).
    """
    kthFile = os.path.join(config.shared_sick, str(id), 'kth.mod')
    if not os.path.isfile(kthFile):
        return 0
    kth = get_number_of_instances(kthFile)
    kt = get_number_of_instances(os.path.join(config.shared_sick, str(id), 'kt.mod'))
    kh = get_number_of_instances(os.path.join(config.shared_sick, str(id), 'kh.mod'))
    if kh == 0:
        return 0
    return 1 - (kth - kt) / kh
        
def instance_overlap2(id, sentence_a, sentence_b):
    """
    Calculate the amount of overlap between the instances in the models of sentence_a (t) and sentence_b (h).
    And also try to do the same while replacing words with paraphrases to obtain a higher score.
    """
    score = 0
    kthFile = os.path.join(config.shared_sick, str(id), 'kth.mod')
    if not os.path.isfile(kthFile):
        score = 0
    else:
        kth = get_number_of_instances(kthFile)
        kt = get_number_of_instances(os.path.join(config.shared_sick, str(id), 'kt.mod'))
        kh = get_number_of_instances(os.path.join(config.shared_sick, str(id), 'kh.mod'))
        if kh == 0:
            score = 0
        else: 
            score = 1 - (kth - kt) / kh

    for counter in range(1,8):
        newfolder = os.path.join(config.shared_sick2,'{0}.{1}'.format(str(id), counter))
        if os.path.isfile(os.path.join(newfolder, 'kth.mod')):
            kth = get_number_of_instances(os.path.join(newfolder, 'kth.mod'))
            kt = get_number_of_instances(os.path.join(newfolder, 'kt.mod'))
            kh = get_number_of_instances(os.path.join(newfolder, 'kh.mod'))
            if kh > 0:
                new_score = 1 - (kth - kt) / kh
                if new_score > score:
                    score = new_score
    return score

def get_number_of_relations(modelfile):
    """
    Return the amount of relations in the modelfile.
    """
    counter = 0
    for line in open(modelfile):
        if line.find('f(2') > 0:
            counter += line.count('(')-1
    return float(counter)
  
def relation_overlap(id, sentence_a, sentence_b):
    """
    Calculate the amount of overlap between the relations in the models of sentence_a (t) and sentence_b (h).
    """
    kthFile = os.path.join(config.shared_sick, str(id), 'kth.mod')
    if not os.path.isfile(kthFile):
        return 0
    kth = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kth.mod'))
    kt = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kt.mod'))
    kh = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kh.mod'))
    if kh == 0:
        return 0
    return 1 - (kth -kt) / kh
  
def relation_overlap2(id, sentence_a, sentence_b):
    """
    Calculate the amount of overlap between the relations in the models of sentence_a (t) and sentence_b (h).
    And also try to do the same while replacing words with paraphrases to obtain a higher score.
    """
    score = 0
    kthFile = os.path.join(config.shared_sick, str(id), 'kth.mod')
    if not os.path.isfile(kthFile):
        score = 0
    else:
        kth = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kth.mod'))
        kt = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kt.mod'))
        kh = get_number_of_relations(os.path.join(config.shared_sick, str(id), 'kh.mod'))
        if kh == 0:
            score = 0
        else:
            score = 1 - (kth -kt) / kh
    
    for counter in range(1,8):
        newfolder = os.path.join(config.shared_sick2,'{0}.{1}'.format(str(id), counter))
        if os.path.isfile(os.path.join(newfolder, 'kth.mod')):
            kth = get_number_of_relations(os.path.join(newfolder, 'kth.mod'))
            kt = get_number_of_relations(os.path.join(newfolder, 'kt.mod'))
            kh = get_number_of_relations(os.path.join(newfolder, 'kh.mod'))
            if kh > 0:
                new_score = 1 - (kth - kt) / kh
                if new_score > score:
                    score = new_score
    return score
  
def word_overlap2(sentence_a, sentence_b):
    """
    Calculate the word overlap of two sentences and tries to use paraphrases to get a higher score
    """
    
    a_set = set(word for word in sentence_a) - config.stop_list
    b_set = set(word for word in sentence_b) - config.stop_list
    score = len(a_set&b_set)/float(len(a_set|b_set))

    for replacement in get_replacements(sentence_a, sentence_b):
        sentence_a[sentence_a.index(replacement[0])] = replacement[1]
        a_set = set(word for word in sentence_a) - config.stop_list
        b_set = set(word for word in sentence_b) - config.stop_list
        newScore = len(a_set&b_set)/float(len(a_set|b_set))
        if newScore > score:
            score = newScore
        sentence_a[sentence_a.index(replacement[1])] = replacement[0]
    return score
  
def get_entailment_judgements():
    """
    Get entailment judgements from Johan's system,
    return as a dict mapping to a list with the appropriate index set to 1.
    """
    results = defaultdict(lambda: [0,0,0])
    mapping = dict(zip(('CONTRADICTION','ENTAILMENT','NEUTRAL'), range(3)))

    for line in open('working/results.raw'):
        words = line.split()
        sick_id = str(words[0])
        result = words[-1]

        # Set the index correspoinding to the judgement to 1
        results[sick_id][mapping[result]] = 1

    return results
      
def bigrams(sentence):
    """
    Since the skipgram model includes bigrams, look for them.
    These are represented as word1_word2.
    """
    return [word+'_'+sentence[i+1] 
            if word+'_'+sentence[i+1] in word_ids else None 
                for i, word in enumerate(sentence[:-1])] if config.USE_BIGRAMS else []

def trigrams(sentence):
    """
    Since the skipgram model includes trigrams, look for them.
    These are represented as word1_word2_word3.
    """
    return [word+'_'+sentence[i+1]+'_'+sentence[i+2] 
            if word+'_'+sentence[i+1]+'_'+sentence[i+2] in word_ids else None 
                for i, word in enumerate(sentence[:-2])] if config.USE_TRIGRAMS else []
        
def sentence_composition(sentence_a, sentence_b):
    """
    Return the composition of two sentences (element-wise multiplication)
    """
    sent_a = np.sum([projections[word_ids[word]] 
        if word in word_ids else [0] 
            for word in sentence_a+bigrams(sentence_a)], axis=0)
    sent_b = np.sum([projections[word_ids[word]] 
        if word in word_ids else [0] 
            for word in sentence_b+bigrams(sentence_b)], axis=0)
    
    reps = sent_a * sent_b

    return reps

def sentence_distance(sentence_a, sentence_b):
    """
    Return the cosine distance between two sentences
    """
    
    sent_a = np.sum([projections[word_ids[word]] 
        if word in word_ids else [0] 
            for word in sentence_a+bigrams(sentence_a)+trigrams(sentence_a)], axis=0)
    sent_b = np.sum([projections[word_ids[word]] 
        if word in word_ids else [0] 
            for word in sentence_b+bigrams(sentence_b)+trigrams(sentence_b)], axis=0)
   
    return cosine(sent_a, sent_b)

def synset_overlap(sentence_a, sentence_b):
    """
    Calculate the synset overlap of two sentences.
    Currently uses the first 5 noun senses.
    """
    def synsets(word):
        sense_lemmas = []
        for pos in ('n'):#,'a'):
            for i in xrange(5):
                try:
                    sense_lemmas += [lemma.name 
                        for lemma in wn.synset('{0}.{1}.0{2}'.format(word, pos, i)).lemmas]
                except WordNetError: 
                    pass
        return sense_lemmas

    a_set = set(lemma for word in sentence_a for lemma in synsets(word))
    b_set = set(lemma for word in sentence_b for lemma in synsets(word))
    score = len(a_set&b_set)/float(len(a_set|b_set))

    return score

def synset_distance(sentence_a, sentence_b):
    def distance(word, sentence_b):
        try:
            synset_a = wn.synset('{0}.n.01'.format(word))
        except WordNetError:
            return 0.0

        max_similarity = 0.0
        for word2 in sentence_b:
            try:
                similarity = synset_a.path_similarity(wn.synset('{0}.n.01'.format(word2)))
                if similarity > max_similarity:
                    max_similarity = similarity
            except WordNetError:
                continue

        return max_similarity

    distances = [distance(word, sentence_b) for word in sentence_a]

    return sum(distances)/float(len([1 for i in distances if i > 0.0]))

def word_overlap(sentence_a, sentence_b):
    """
    Calculate the word overlap of two sentences.
    """
    a_set = set(word for word in sentence_a) - config.stop_list
    b_set = set(word for word in sentence_b) - config.stop_list
    score = len(a_set&b_set)/float(len(a_set|b_set))# len(s1&s2)/max(len(s1),len(s2))

    return score

def sentence_lengths(sentence_a, sentence_b):
    """
    Calculate the proportionate difference in sentence lengths.
    """
    return abs(len(sentence_a)-len(sentence_b))/float(min(len(sentence_a),len(sentence_b)))

url = 'http://127.0.0.1:7777/raw/pipeline?format=xml'
def sent_complexity(sentence):
    r = requests.post(url, data=' '.join(sentence))
    complexity = drs_complexity.parse_xml(r.text)

    print complexity
    return complexity

def drs_complexity_difference(sentence_a, sentence_b):
    sent_a_complexity = sent_complexity(sentence_a)
    sent_b_complexity = sent_complexity(sentence_b)

    return abs(sent_a_complexity-sent_b_complexity)

if config.RECALC_FEATURES:
    # Load projection data
    word_ids, projections = load_semeval_data.load_embeddings()

    entailment_judgements = get_entailment_judgements()

    # Load the paraphrases data.
    paraphrases = {}
    for line in open(config.ppdb):
        source = line.split('|')[1][1:-1]
        target = line.split('|')[4][1:-1]
        if source in paraphrases:
            paraphrases[source].append(target)
        else:
            paraphrases[source] = [target]