#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

'''
Args: Bubblesort observations file, Insertion sort observations file
Valid Values :
Sample execution:
python plot-sort-results.py bubble-points.txt insertion-points.txt
'''

def read_file(file_name):
    print("Reading File ", file_name)
    points_dict = {}
    f = open(file_name, 'r')
    for line in iter(f):
        this_line = line.rstrip('\n')
        if len(this_line) != 0:
            tokens = this_line.split(',')
            points_dict[int(tokens[0])] = int(tokens[1])
    f.close()
    #Return sorted dict by key
    return OrderedDict(sorted(points_dict.items()))


def plot_bar_charts(bubble_points_dict, insertion_points_dict):
    print("In plot_bar_charts() method")

    # data to plot
    n_groups = len(bubble_points_dict)
    bubble_values = tuple(bubble_points_dict.values())
    insertion_values = tuple(insertion_points_dict.values())
    #print type(bubble_values)
    #print bubble_values

    xticks = tuple(insertion_points_dict.keys())
    print xticks

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, bubble_values, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Bubble')
    rects2 = plt.bar(index + bar_width, insertion_values, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Insertion')

    plt.xlabel('Seed Count')
    plt.ylabel('Execution Time')
    plt.title('Bubble Sort Vs Insertion Sort')
    plt.xticks(index + bar_width, xticks)
    plt.legend()

    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Plot time taken to sort an array of different sizes")

    bubble_points_file = sys.argv[1]
    insertion_points_file = sys.argv[2]

    bubble_points_dict = read_file(bubble_points_file)
    insertion_points_dict = read_file(insertion_points_file)

    print bubble_points_dict
    print insertion_points_dict

    print("bubble_points_dict point count    : ", len(bubble_points_dict))
    print("insertion_points_dict point count : ", len(insertion_points_dict))

    if len(bubble_points_dict) != len(insertion_points_dict):
        print("Observations counts in two dataset DOES NOT match. Will not plot")
    else:
        print("Observations counts in two dataset match. Will plot")
        plot_bar_charts(bubble_points_dict, insertion_points_dict)

    print("End of program.")


