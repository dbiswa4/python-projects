import sys
import pandas as pd
import matplotlib.pyplot as plt

def draw_linechart(filename):
    lines = pd.read_csv(filename)
    x = lines['year']
    y = lines['counts']

    plt.plot(x,y)
    plt.title('Aircrash Statistics : 1908-2009', color='black')
    plt.xlabel('Year')
    plt.ylabel('Number of Aircrash')
    plt.savefig('linechart.png')

if __name__ == '__main__':
    print 'Executing linechart'

    filename = sys.argv[1]
    draw_linechart(filename)