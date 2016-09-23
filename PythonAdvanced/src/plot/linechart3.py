import sys
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import pylab


def data_cleanup(filename):
    airplane = pd.read_csv(filename, parse_dates=[0])
    print airplane.columns

    airplane_clean = airplane.fillna(0)

    return airplane_clean

def linechart_data_prep(filename):
    airplane_clean= data_cleanup(filename)
    airplane_clean['year'] = airplane_clean['Date'].dt.year
    accident_per_year = pd.DataFrame(airplane_clean.groupby('year').size().rename('counts'))
    draw_linechart(accident_per_year, 'Aircrash History')

def draw_linechart(df, title):
    f = plt.figure(figsize=(10, 10))
    df.plot(ax=f.gca())
    plt.title(title, color='black')

    plt.plot(df)
    plt.show()


if __name__ == '__main__':
    print 'I am inside main'
    linechart_data_prep('Airplane_Crashes_and_Fatalities_Since_1908.csv')
