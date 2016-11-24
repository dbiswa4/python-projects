import sys
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import pylab
import random
import numpy as np


def random_data_gen(filename):
    caravan_dataset = pd.read_csv(filename)
    zero_data_rows = caravan_dataset[caravan_dataset['CARAVAN'] == 0]
    one_data_rows = caravan_dataset[caravan_dataset['CARAVAN'] == 1]
    rows = np.random.choice(zero_data_rows.index.values, 500)
    zero_random_df = zero_data_rows.ix[rows]
    result_df = one_data_rows.append(zero_random_df)
    result_df.to_csv('ticdata2000_train_1k-5.csv', index = False)
    #result_df.to_csv('ticeval2000_test_1k-5.csv', index = False)


if __name__ == '__main__':
    print 'This is my Random row selection program'
    random_data_gen('ticdata2000_train.csv')
    #random_data_gen('ticeval2000_test.csv')