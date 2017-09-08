#!/usr/bin/env python

import sys
import random

'''
Args: Number of seeds, plot condition
Valid Values : Plot Condition - 1,2,3
Sample execution:
python gen-input.py 10 1
'''


def plot1_file(n=2000):
    print("Generate Input for Plot1")
    file_name = "plot1-" + str(n) + ".txt"

    with open(file_name, 'w+') as f:
        for i in range(1,n+1):
            num = random.randint(1,n)
            f.write("%s\n" % num)
    f.close()

    print("Done generating file")

def plot2_file(n=2000):
    print("Generate Input for Plot2")


def plot3_file(n=2000):
    print("Generate Input for Plot3")

if __name__ == '__main__':
    print("Generate Input")

    n = int(sys.argv[1])
    plot = int(sys.argv[2])

    if plot == 1:
        plot1_file(n)
    elif plot == 2:
        plot2_file(n)
    elif plot == 3:
        plot3_file(n)
    else:
        print("Not a valid option")