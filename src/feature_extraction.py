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


def word_overlap(sentence_a, sentence_b):
    """
    Calculate the word overlap of two sentences.
    """
    a_set = set(word for word in sentence_a) - config.stop_list
    b_set = set(word for word in sentence_b) - config.stop_list
    score = len(a_set&b_set)/float(len(a_set|b_set))# len(s1&s2)/max(len(s1),len(s2))

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

def get_number_of_instances(model):
    if model is None:
        return 0
    else:
        return float(len(model[0].split('d'))-2)

def get_instance_overlap(kt_mod, kh_mod, kth_mod):
    kt = get_number_of_instances(kt_mod)
    kh = get_number_of_instances(kh_mod)
    kth = get_number_of_instances(kth_mod)
    if kh == 0 or kt == 0 or kth == 0:
        return 0
    else: 
        return 1 - (kth - kt) / kh
    
def instance_overlap(kt_mod, kh_mod, kth_mod, replacements):
    """
    Calculate the amount of overlap between the instances in the models of sentence_a (t) and sentence_b (h).
    And also try to do the same while replacing words with paraphrases to obtain a higher score.
    """
    score = get_instance_overlap(kt_mod, kh_mod, kth_mod)
    for replacement in replacements:
        new_score = get_instance_overlap(replacement[5], replacement[6], replacement[7])
        if new_score > score:
            score = new_score
    return score

def get_number_of_relations(model):
    """
    Return the amount of relations in the modelfile.
    """
    if model == None:
        return 0
    counter = 0
    for line in model:
        if line.find('f(2') >= 0:
            counter += 1
    return float(counter)
    # ALS MEERDERE DEZELFDE RELATIE HEBBEN TELT DIT ALS 1!!!!
    
def get_relation_overlap(kt_mod, kh_mod, kth_mod):
    kt = get_number_of_relations(kt_mod)
    kh = get_number_of_relations(kh_mod)
    kth = get_number_of_relations(kth_mod)
    if kh == 0 or kt == 0 or kth == 0:
        return 0
    else:
        return 1 - (kth - kt) / kh
  
def relation_overlap(kt_mod, kh_mod, kth_mod, replacements):
    """
    Calculate the amount of overlap between the relations in the models of sentence_a (t) and sentence_b (h).
    """
    score = get_relation_overlap(kt_mod, kh_mod, kth_mod)
    for replacement in replacements:
        new_score = get_relation_overlap(replacement[5], replacement[6], replacement[7])
        if new_score > score:
            print(score, new_score)
            score = new_score
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


def get_shared_features(pair_id, sentence_a, sentence_b):
    root = './working/sick/'+pair_id
    data = ()
    with open(root+'/prediction.txt', 'r') as in_f:
        line = in_f.readline().lower().strip()
        data = data + (prediction_ids[line],)

    with open(root+'/modsizedif.txt', 'r') as in_f:
        lines = in_f.readlines()
        prover  = prover_ids[lines[0].split()[0][:-1]] # First field of line without trailing '.'
        dom_nov = float(lines[1].split()[0][:-1]) 
        rel_nov = float(lines[2].split()[0][:-1])
        wrd_nov = float(lines[3].split()[0][:-1])
        mod_nov = float(lines[4].split()[0][:-1])
        word_overlap = float(lines[5].split()[0][:-1])

        data = data + (prover, dom_nov, rel_nov, wrd_nov, mod_nov, word_overlap)

    with open(root+'/complexities.txt', 'r') as in_f:
        line = in_f.readlines()
        data = data + (float(line[0]), float(line[1]))

    return data

if config.RECALC_FEATURES:
    # Load projection data
    word_ids, projections = load_semeval_data.load_embeddings()

    entailment_judgements = get_entailment_judgements()