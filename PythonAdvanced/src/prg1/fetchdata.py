import pandas as pd

if __name__ == '__main__':
    print 'Execute the cleanup process'

    filename = 'Airplane_Crashes_and_Fatalities_Since_1908.csv'
    airplane = pd.read_csv(filename, parse_dates=[0])
    #print airplane.columns

    #Remove null from data
    airplane_clean = airplane.fillna(0)
    #Year field is added
    airplane_clean['year'] = airplane_clean['Date'].dt.year

    #Prepare the dataset for line chart
    accident_per_year = pd.DataFrame(airplane_clean.groupby('year').size().rename('counts'))
    #print accident_per_year.head()
    accident_per_year.to_csv('data-line.csv')

    #Prepare dataset for Histogram
    fatalities = airplane_clean[['year', 'Fatalities']]
    fatalities_df = pd.DataFrame(fatalities)
    fatalities_df.to_csv('data-hist.csv', index=False)


    # Prepare dataset for Barchart
    fatalities_grp = fatalities.groupby('year')
    #print fatalities_grp.head()
    fatalities_sum = fatalities_grp.sum()
    #print fatalities_sum
    fatalities_sum_df = pd.DataFrame(fatalities_sum)
    fatalities_sum_df.to_csv('data-bar.csv')










