# For usage information, type:

# python compute_overlap_baseline.py -h

# The script computes a word overlap baseline for sentence pairs in
# the SEMEVAL 2014 TASK 1 format. The input file must have a header
# line (ignored) followed by rows with tab-delimited fields. The first
# field is an ID, the second and third contain the sentences to be
# compared. Any other field is ignored.

# The output lines (sent to standard output) contain 3 tab-delimited
# fields, pertaining to the pairs in the corresponding input lines: 1)
# the ID, 2) an entailment response (ENTAILMENT or NEUTRAL or
# CONTRADICTION) and 3) the overlap score (that is also considered as
# the script "similarity judgment"). The first output line is a header
# containing the tab-delimited strings pair_ID, entailment_judgment
# and relatedness_score, as required by the evaluation script.

# Overlap is computed as the number of distinct words (or more
# generally tokens, since punctuation marks are also considered) in
# both sentences divided by the number of distinct words in the longer
# one.

# Before computing overlap, stop words are stripped off both
# sentences. The stop words are simply the N most frequent words
# (tokens) in the input, where N can be passed by the user with option
# -s (see comments on the -s option below for the current default
# value, and how this default was determined).

# The user can also pass two thresholds to control the entailment
# responses: the one passed with -c determines the maximum overlap
# value for which a pair will be classified as CONTRADICTION; the one
# passed with -e determines the minimum overlap value for which a pair
# is classified s ENTAILMENT. Any pair with values between these
# thresholds will be classified as NEUTRAL (see comments on the
# relevant options below for their defaults and how they were
# determined).

# This script is free software. You may copy or redistribute it under
# the same terms as Python itself.

from __future__ import division
import re
import argparse
import sys
from collections import deque

# input parameters processed here
parser = argparse.ArgumentParser(description="This script computes a normalized word overlap score and related entailment prediction for each sentence pair of a file in SEMEVAL 2014 TASK 1 format: please look at the script source code for detailed information.")

## INPUT FILE
## We expect one input file in SEMEVAL 2014 TASK 1 format. More
## specifically, we expect the input file to have one header line (to
## be ignored), and each line to begin with 3 tab-delimited fields
## containing 1) pair ID, 2) first sentence, 3) second
## sentence. Everything else is ignored.
parser.add_argument("filename", help="file in SEMEVAL 2014 TASK 1 format")

## STOP WORD COUNT
## User can specify, with option -s, the number of most frequent words
## (or tokens, as this will also include any punctuation marks) to be
## ignored (i.e., to be treated as "stop words") when measuring
## overlap. If the value is 0, no token is considered a stop word; the
## default was optimized on the task of Pearson correlation with trial
## and train similarity judgments, considering possible values from 0
## to 300.
parser.add_argument("--stop", "-s", help="number of most frequent stop words to be ignored (can be 0, default is 3)", type=int, default=3)

## THRESHOLD FOR CONTRADICTION
## User can specify with option -c a threshold (on 0-1 scale) for
## maximum overlap value at which a sentence pair is hypothesized to
## be in the contradiction relation: this value must be lower than the
## entailment threshold (see next)! Default was determined by
## maximizing accuracy on trial and train entailment data, after
## fixing the stop word count value on the similarity data as
## described above (we explored values from 0 to 0.99 in 0.01 steps).
parser.add_argument("--contradiction", "-c", help="maximum score for pair in contradiction relation (can range from 0 to 1 but must be below entailment score, default is 0.03)", type=float, default=0.03)

## THRESHOLD FOR ENTAILMENT
## User can specify with option -e a threshold (on 0-1 scale) for
## minimum overlap value at which a sentence pair is hypothesized to
## be in the entailment relation: this value must be higher than the
## contradiction threshold (see above)! Default was determined by
## maximizing accuracy on trial and train entailment data, after
## fixing the stop word count value on the similarity data as
## described above (we explored values from 0.01 to 1 in 0.01 steps,
## with the constraint that they had to be above the corresponding
## contradiction thresholds).
parser.add_argument("--entailment", "-e", help="minimum score for pair in entailment relation (can range from 0 to 1 but must be above contradiction score, default is 0.56)", type=float, default=0.56)

args = parser.parse_args()

# checking that stop word number is not negative
if (args.stop<0):
    sys.exit("you can't remove a negative number of stop words!")

# checking ranges of contradiction and entailment thresholds
if (args.contradiction<0)or(args.contradiction>1):
    sys.exit("-c threshold must be in 0-1 range!")
if (args.entailment<0)or(args.entailment>1):
    sys.exit("-e threshold must be in 0-1 range!")
if (args.contradiction>=args.entailment):
    sys.exit("contradiction threshold must be below entailment threshold!")


# global lists defined here

## a queue (first in, first out) to store the pair ids in the original
## order
pair_id_queue = deque()
## a list to store the sentence pairs as sets
sentence_set_pair_list = []
## a dictionary to store word frequencies
word_freqs = {}
## the stop word set is initialized to the empty set
stop_word_set = set()

# read input data in SICK training format: expect tab-delimited
# fields, with sentence id in field 0 and the two sentences in field 1
# and 2
with open(args.filename) as data:
    next(data) # skip header line!
    for line in data:
        fields = line.rstrip('[\n\r]+').split("\t")
        # store id
        pair_id_queue.append(fields[0])
        curr_sent_pair = []
        # count words in sentences (that are in the second and third field of input)
        for s in fields[1:3]:
            # simple space-based tokenization, treating
            # character+digit strings and punctuation marks as
            # separated tokens
            tokens = re.findall(r"[\w]+|[^\s\w]+",s.lower())
            # store the set of distinct tokens in sentence for later
            # comparison with the other sentence in pair
            curr_sent_pair.append(set(tokens))
            # counting the tokens to determine stop words
            for t in tokens:
                word_freqs[t] = word_freqs.get(t, 0) + 1
            # done traversing tokens
        # done traversing sentence pair
        # we add sentence pair to a sentence pair list
        sentence_set_pair_list.append(curr_sent_pair)
    # done traversing input file
# done opening input file

# the stop word count must be below the total word (= distinct token)
# count!
if args.stop>=len(word_freqs):
    sys.exit("stop word count cannot be higher or equal to distinct word count in dataset!")

# determining stop words: sort frequency list by freq, and put the
# top args.stop in a set:
if args.stop>0:
    stop_word_set=set(sorted(word_freqs,key=word_freqs.get,reverse=True)[:args.stop])

# now we traverse the sentence pair list, and consider the
# intersection after removing the stop words

# first, print header as required by Task 1 evaluation
print "pair_ID\tentailment_judgment\trelatedness_score"

for s1,s2 in sentence_set_pair_list:
    s1 = s1 - stop_word_set
    s2 = s2 - stop_word_set

    # word overlap computed as intersection between s1 and s2, normalized
    # by cardinality of more-populated set
    ## sanity!: if after stop word removal both sentence sets are empty,
    ## overlap is taken to be 1!!!
    if max(len(s1),len(s2))==0:
        overlap=1
    else:
        overlap=len(s1&s2)/max(len(s1),len(s2))

    # for the entailment task, we return ENTAILMENT if overlap score>=e
    # threshold, else NEUTRAL if overlap score >c, CONTRADICTION
    # otherwise
    entailment_response = "CONTRADICTION"
    if overlap>args.contradiction:
        entailment_response = "NEUTRAL"
    if overlap>=args.entailment:
        entailment_response = "ENTAILMENT"
  
    # as per the header column order, print pair id, entailment
    # judgment and relatedness score (that is actually the same as
    # overlap!)
    output_string = pair_id_queue.popleft() + "\t" + entailment_response + "\t" + str(overlap)
    print output_string

# debug: print more info
#    print pair_id_queue.popleft(),s1,s2,len(s1&s2),max(len(s1),len(s2)), overlap,entailment_response

