import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_hist(filename, binwidth = 25):
    lines = pd.read_csv(filename)

    #Since we are analysising the Fatalities, we are dropping the records where there hs been no casualties
    lines = lines.drop(lines['Fatalities'] == 0.0)

    x = lines['year']
    y = lines['Fatalities']

    print y.max()
    print y.min()
    print y.count()

    plt.hist(y, bins=np.arange(min(y), max(y) + binwidth, binwidth))

    plt.title('Aircrash Fatalities vs Number of Aircrash', color='black')
    plt.xlabel('Number of Casualties')
    plt.ylabel('Number of Aircrash')
    #plt.show()
    plt.savefig('histogram.png')

if __name__ == '__main__':
    print 'Executing Histogram script'

    filename = sys.argv[1]
    draw_hist(filename)