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


def word_overlap2(sentence_a, sentence_b):
    """
    Calculate the word overlap of two sentences.
    """
    a_set = set(word for word in sentence_a) - config.stop_list
    b_set = set(word for word in sentence_b) - config.stop_list
    score = len(a_set&b_set)/float(len(a_set|b_set))# len(s1&s2)/max(len(s1),len(s2))

    return score

def word_overlap3(t_raw, h_raw, replacements):
    """
    Calculate the word overlap of two sentences and tries to use paraphrases to get a higher score
    """
    t_set = set(word for word in t_raw) - config.stop_list
    h_set = set(word for word in h_raw) - config.stop_list
    score = len(t_set & h_set) / float(len(t_set|h_set))

    for replacement in replacements:
        t_set = set(word for word in replacement[2]) - config.stop_list # replacement[1] = t_raw
        h_set = set(word for word in replacement[3]) - config.stop_list # replacement[2] = h_raw
        newScore = len(t_set & h_set) / float(len(t_set|h_set))
        if newScore > score:
            score = newScore
    return score


def sentence_lengths(sentence_a, sentence_b):
    """
    Calculate the proportionate difference in sentence lengths.
    """
    return abs(len(sentence_a)-len(sentence_b))/float(min(len(sentence_a),len(sentence_b)))
      
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
   
    return float(cosine(sent_a, sent_b))

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
#TODO use replacements?

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
#TODO use replacements?

def get_number_of_instances(model):
    """
    Return the number of instances in the model
    """
    if model is None:
        return 0
    else:
        return float(len(model[0].split('d'))-2)

def get_instance_overlap(kt_mod, kh_mod, kth_mod):
    """
    Calculate the amount of overlap using the number of instance overlap
    """
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
        new_score = get_instance_overlap(replacement[6], replacement[7], replacement[8])
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
    #TODO  when multiples of same relation, the result is still 1
    
def get_relation_overlap(kt_mod, kh_mod, kth_mod):
    """
    Calculate the amount of overlap using the number of relations
    """
    kt = get_number_of_relations(kt_mod)
    kh = get_number_of_relations(kh_mod)
    kth = get_number_of_relations(kth_mod)
    if kh == 0 or kt == 0 or kth == 0:
        return 0
    else:
        return 1 - (kth - kt) / kh
  
def relation_overlap(kt_mod, kh_mod, kth_mod, replacements):
    """
    Calculate the amount of overlap between the relations in the models of t and h.
    """
    score = get_relation_overlap(kt_mod, kh_mod, kth_mod)
    for replacement in replacements:
        new_score = get_relation_overlap(replacement[6], replacement[7], replacement[8])
        if new_score > score:
            score = new_score
    return score

def get_nouns(root):
    """
    Return the list of nouns as found in the boxer xml 'root'
    """
    nouns = []
    for child in root.findall("./xdrs/taggedtokens/tagtoken/tags"):
        noun = False
        for grandchildren in child.findall("./tag[@type='pos']"):
            if grandchildren.text == 'NN' or grandchildren.text == 'NNS':
                noun = True
        if noun == True:
            for grandchildren in child.findall("./tag[@type='lemma']"):
                nouns.append(grandchildren.text)
    return nouns

def noun_overlap(t_xml, h_xml, replacements):
    """
    Calculate the amount of overlap between all nouns in t and h
    """
    score = 0
    if t_xml == None or h_xml == None:
        return 0
    t_set = set(get_nouns(t_xml.getroot()))
    h_set = set(get_nouns(h_xml.getroot()))
    if float(len(t_set | h_set)) > 0:
        score = len(t_set & h_set) / float(len(t_set | h_set))
    '''
    for replacement in replacements:
        t_set = set(get_nouns(replacements[8].getroot())) ##waarom werkt dit niet?
        h_set = set(get_nouns(replacements[9].getroot()))
        if float(len(t_set | h_set)) > 0:
            new_score = len(t_set & h_set) / float(len(t_set | h_set))
            if new_score > score:
                score = new_score
    '''
    return score
    
def get_verbs(root):
    """
    Return the list of verbs as found in the boxer xml 'root'
    """
    verbs = []
    for child in root.findall("./xdrs/taggedtokens/tagtoken/tags"):
        noun = False
        for grandchildren in child.findall("./tag[@type='pos']"):
            if grandchildren.text == 'VBP' or grandchildren.text == 'VBG':
                noun = True
        if noun == True:
            for grandchildren in child.findall("./tag[@type='lemma']"):
                verbs.append(grandchildren.text)
    return verbs

def verb_overlap(t_xml, h_xml, replacements):
    """
    Calculate the amount of overlap between all verbs in t and h
    """
    score = 0
    if t_xml == None or h_xml == None:
        return 0
    t_set = set(get_verbs(t_xml.getroot()))
    h_set = set(get_verbs(h_xml.getroot()))
    if float(len(t_set | h_set)) > 0:
        score = len(t_set & h_set) / float(len(t_set | h_set))
    '''
    for replacement in replacements:
        t_set = set(get_verbs(replacements[8].getroot()))
        h_set = set(get_verbs(replacements[9].getroot()))
        if float(len(t_set | h_set)) > 0:
            new_score = len(t_set & h_set) / float(len(t_set | h_set))
            if new_score > score:
                score = new_score
    '''
    return score

# Used to encode the entailment judgements numerically
prediction_ids = defaultdict(lambda:len(prediction_ids))
prover_ids = defaultdict(lambda:len(prover_ids))

def get_johans_features(modsizedif, prediction):
    """
    Read the outputs of johans system
    """
    data = []
    
    prover_output = 0
    
    if modsizedif[0].split()[0] == 'contradiction.':
        prover_output = 0.0
    if modsizedif[0].split()[0] == 'unknown.':
        prover_output = 0.5
    if modsizedif[0].split()[0] == 'proof.':
        prover_output = 1.0

    data.append(prover_output) # prover output
    data.append(float(modsizedif[1].split()[0][:-1]))      # domain novelty
    data.append(float(modsizedif[2].split()[0][:-1]))      # relation novelty
    data.append(float(modsizedif[3].split()[0][:-1]))      # wordnet novelty
    data.append(float(modsizedif[4].split()[0][:-1]))      # model novelty
    data.append(float(modsizedif[5].split()[0][:-1]))      # word overlap
    
    if prediction[0].split()[0] == 'informative':          # prediction.txt
        data.append(0)
    else:
        data.append(1)

    return data
#TODO, also use sick2?

def get_entailment_judgements():
    """
Get entailment judgements from Johan's system,
return as a dict mapping to a list with the appropriate index set to 1.
"""
    results = defaultdict(lambda: [0,0,0])
    mapping = dict(zip(('CONTRADICTION','ENTAILMENT','NEUTRAL'), range(3)))

    for line in open('working/sick.run'):
        words = line.split()
        sick_id = str(words[0])
        result = words[1]

        # Set the index correspoinding to the judgement to 1
        results[sick_id][mapping[result]] = 1

    return results


############################################################


url = 'http://127.0.0.1:7777/raw/pipeline?format=xml'
def sent_complexity(sentence):
    r = requests.post(url, data=' '.join(sentence))
    complexity = drs_complexity.parse_xml(r.text)

    return complexity

def drs_complexity_difference(sentence_a, sentence_b):
    sent_a_complexity = sent_complexity(sentence_a)
    sent_b_complexity = sent_complexity(sentence_b)

    return abs(sent_a_complexity-sent_b_complexity)




if config.RECALC_FEATURES:
    # Load projection data
    word_ids, projections = load_semeval_data.load_embeddings()

    entailment_judgements = get_entailment_judgements()