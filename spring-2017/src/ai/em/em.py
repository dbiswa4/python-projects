# Candy2.py
# by ___________
# based on skeleton code by D. Crandall, 11/2016
#
# A candy manufacturer makes 5 different types of candy bags, each of which 
# is filled with lime and cherry candies but in different proportions. 
# 
# We've bought 100 bags chosen at random. For each bag, we've then opened 
# it, randomly drawn 100 candies, and recorded the flavor of each one.
#
# We want to estimate (1) what's the actual percentage of cherry candies
# in each of the 5 bag types, and (2) what's the actual bag type of each
# of our 100 bags?

import random
import numpy as np
import math
import copy

#####
# You shouldn't have to modify this part

# These are the *actual* values of C0 ... C4 we're trying to estimate.
# Shh... They're a secret! :) This is what we're trying to estimate.
bagtype_count = 5
actual_cs = (0.2, 0.3, 0.7, 0.9, 1.0)

# Now sample 100 bags
bag_count = 100
actual_bagtypes = [ random.randrange(0, bagtype_count) for i in range(0, bag_count) ]

# Now sample 100 candies from each bag, to produce a list-of-lists
candy_count = 100
observations = [ [ ("L", "C")[x] for x in tuple(np.random.binomial( 1, actual_cs[ bagtype ], candy_count ) ) ] for bagtype in actual_bagtypes ]

######
# This is the part you'll want to edit

# This list will hold your estimated C0 ... C4 values, and your estimated
# bagtype for each bag.
estimated_cs = [0] * bagtype_count
estimated_bagtypes = [0] * bag_count

# Here's pseudocode for what you should implement:
#
#
# Run EM multiple times:
#
#   Randomly initialize estimated_cs
#   Until estimated probabilities converge:
#
#      # E-step
#      For each sampled bag:
#          Calculate probability of the data given each model, i.e. of the observed candies in this bag assuming each of the 5 bagtypes
#          Put the highest-probability bagtype into estimated_labels[bag]
#
#      # M-step
#      For each bagtype:
#          Estimate probability c_i for this bagtype i using the bags currently assigned to this bagtype in estimated_labels
#          Update estimated_cs[bagtype]
# 
#   Calculate probability (or log-probability) of the data given the final values of estimated_cs
#
# Select the model with the highest probability of the data given estimated_cs

#Model Implementation
prev_estimated_cs = [0] * bagtype_count
estimated_cs = np.random.uniform(0.0,1.0,5).tolist()

#Assumption:
#If result does not converse in below max iteration, loop is exited
max_itr = 2000
itr = 0
def is_converged():
    print("Current Estimation          : ", estimated_cs)
    print("Previous Estimation         : ", prev_estimated_cs)
    improvement = max([abs(cs_i - prev_cs_i) for cs_i, prev_cs_i in zip(estimated_cs, prev_estimated_cs)])
    if improvement > 0.01 and itr < max_itr:
        return False
    else:
        return True

while(not is_converged()):
    ind = 0
    itr += 1
    for observation in observations:
        observation_cherry_prob = float(sum([1 for i in observation if i.upper() == 'C'])) / float(len(observations))
        diff = 0.0
        min_diff = float('inf')
        for estimated_prob in estimated_cs:
            diff = abs(estimated_prob - observation_cherry_prob)
            if diff < min_diff:
                min_diff=diff
                temp_bagtype = estimated_cs.index(estimated_prob)
        estimated_bagtypes[ind] = temp_bagtype
        ind += 1

    prev_estimated_cs = copy.deepcopy(estimated_cs)
    for bag_no in range(bagtype_count):
        bag_cherry_prob = float(sum([1 for k in estimated_bagtypes if k == bag_no])) / float(bag_count)
        estimated_cs[bag_no] = bag_cherry_prob


######
# You shouldn't have to modify this part -- it just spits out the results.

# Sort the estimated probabilities so they coincide with the actual ones
#
sorted_cs = sorted((e,i) for i,e in enumerate(estimated_cs))
estimated_cs = [ v[0] for v in sorted_cs ]
index_remapper = [ 0 ] * bagtype_count
for i in range(0, bagtype_count):
    index_remapper[ sorted_cs[i][1] ] = i
estimated_bagtypes = [ index_remapper[bagtype] for bagtype in estimated_bagtypes ]

print ("Actual C's:         ", actual_cs)
print ("Estimated C's:      ", estimated_cs)
print ("Actual bagtypes:    ", actual_bagtypes)
print ("Estimated bagtypes: ", estimated_bagtypes)
print ("Correctly estimated bags: ", sum( [ actual_bagtypes[i] == estimated_bagtypes[i] for i in range(0, len(estimated_bagtypes) ) ] ))