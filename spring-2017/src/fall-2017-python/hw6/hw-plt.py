import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

binsizes = [2, 3, 5, 10, 30, 40, 60, 100 ]

#figsize => w,h tuple in inches
#Figure 1, width is 18 inch, height is 6 inch

'''
movie_df = pd.read_csv('imdb.csv', delimiter='\t')

plt.figure(1, figsize=(18, 8))

print movie_df.size

bins = 40

movie_df['Rating'].hist(bins=bins)

#plt.show()



for i, bins in enumerate(binsizes):
    print i, bins
    for i, bins in enumerate(binsizes):
        # TODO: use subplot and hist() function to draw 8 plots
        plt.subplot(2, 4, i + 1)
        movie_df['Rating'].hist(bins=bins)

'''


data1 = [14, 14, 15, 16, 16, 18, 18, 19, 19, 21, 22, 25, 25, 29, 30, 30]

data2 = [-1, 3, 3, 4, 15, 16, 16, 17, 23, 24, 24, 25, 35, 36, 37, 46]


mpl_fig = plt.figure()



ax = mpl_fig.add_subplot(111)

ax.boxplot(data2)



ax.set_xlabel('Data Points')
ax.set_ylabel('Variance')

plt.show()
