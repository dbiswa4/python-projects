#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

'''
Args: Bubblesort observations file, Insertion sort observations file
Valid Values :
Sample execution:
python plot-sort-results.py bubble-points.txt insertion-points.txt plot1
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


def plot_line_charts(sort1_points_dict, sort2_points_dict, sort3_points_dict, plotname="plot1"):
    print("In plot_bar_charts() method")

    # data to plot
    n_groups = len(sort1_points_dict)
    sort1_values = tuple(sort1_points_dict.values())
    sort2_values = tuple(sort2_points_dict.values())
    sort3_values = tuple(sort3_points_dict.values())

    seed_counts = tuple(sort2_points_dict.keys())
    print seed_counts

    plt.gca().set_color_cycle(['red', 'green', 'blue'])

    plt.plot(seed_counts, sort1_values)
    plt.plot(seed_counts, sort2_values)
    plt.plot(seed_counts, sort3_values)

    plt.xlabel('Number of Elements')
    plt.ylabel('Execution Time in ms')
    plt.title('Sort1 Vs Sort2 Vs Sort3')
    plt.legend(['sort1', 'sort2', 'sort3'], loc='upper left')

    #plt.show()
    plt.savefig(plotname + '.png')


if __name__ == '__main__':
    print("Plot time taken to sort an array of different sizes")

    sort1_points_file = sys.argv[1]
    sort2_points_file = sys.argv[2]
    sort3_points_file = sys.argv[3]
    plotname = sys.argv[4]

    sort1_points_dict = read_file(sort1_points_file)
    sort2_points_dict = read_file(sort2_points_file)
    sort3_points_dict = read_file(sort3_points_file)

    #print sort1_points_dict
    #print sort2_points_dict
    #print sort3_points_dict

    print("sort1_points_dict point count    : ", len(sort1_points_dict))
    print("sort2_points_dict point count : ", len(sort2_points_dict))
    print("sort3_points_dict point count : ", len(sort3_points_dict))

    if len(sort1_points_dict) != len(sort2_points_dict) and len(sort1_points_dict) != len(sort3_points_dict):
        print("Observations counts in two dataset DOES NOT match. Will not plot")
    else:
        print("Observations counts in two dataset match. Will plot")
        #plot_bar_charts(bubble_points_dict, insertion_points_dict)
        plot_line_charts(sort1_points_dict, sort2_points_dict, sort2_points_dict, plotname)

    print("End of program.")


