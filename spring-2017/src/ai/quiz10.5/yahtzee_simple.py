#!/usr/bin/env python

'''
Description:
This program takes 3 arguments. Each of the arguments has to be values from 1 to 6. The values indicates the dice values.

Sample Execution instruction:
python yahtzee_simple.py 1 2 5

'''

import sys
from collections import OrderedDict
from random import randint


def roll_dice():
    return randint(1, 6)

def should_roll(common_val, other_val):
    #The logic can be better
    if common_val >= 4 and other_val >=5:
        print 'Do not roll'
        return False
    return True

def identify_n_roll_dice(states):
    dice_to_roll = []
    if len(set(states.values())) == 2:
        if states[0] == states[1]:
            if should_roll(states[0], states[2]):
                states[2] = roll_dice()
        elif states[1] == states[2]:
            if should_roll(states[1], states[0]):
                states[0] = roll_dice()
        elif states[0] == states[2]:
            if should_roll(states[0], states[1]):
                states[1] = roll_dice()
    else:
        print 'All dices are different'
        for k,v in states.iteritems():
            #We can put a better logic
            if v <= 3:
                dice_to_roll.append(k)
        if not dice_to_roll:
            print 'Roll not recommended'
        else:
            print 'dice_to_roll : ', dice_to_roll
        for x in dice_to_roll:
            print 'Roll ' + str(x) + ' number dice'
            states[x] = roll_dice()

    print 'New dice values :', states.values()
    return states

def is_same_states(states):
    if len(set(states.values())) == 1:
        print 'All the dices have same states'
        return True
    return False

def play_game(states):
    score = 0
    if is_same_states(states):
        score = 25
    else:
        new_states = identify_n_roll_dice(states)
        if is_same_states(new_states):
            score = 25
        else:
            score = sum(new_states.values())

    return score


if __name__ == '__main__':
    print '***Yahtzee Simple Version***'
    if len(sys.argv) != 4:
        print 'Number of arguments are not valid. The program expects 3 arguments - value for each of the three dice...'
        exit(0)
    dice_states = OrderedDict()
    for x in range((0), len(sys.argv)-1):
        #print 'sys.argv[x] : ', sys.argv[x+1]
        dice_states[x]=int(sys.argv[x+1])

    print 'Initial Dice values : ', dice_states.values()
    for k,v in dice_states.iteritems():
        if v == 0:
            print 'One or more values given is/are Zero. Please enter a non-zero value for all dices'
            exit(0)

    score = play_game(dice_states)
    print '\nFinal Score : ', score

