from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



if __name__ == '__main__':
    sparkConf = SparkConf().setAppName("Spark Test")
    sc = SparkContext(conf=sparkConf)
    sqlContext = SQLContext(sc)

    msd = sc.textFile('/Users/dbiswas/Documents/SourceCode/python/python-projects/spring-2017/src/spark2-onramp/MSD_1k.txt')
    print msd.first()

    # Generate some test data
    #x = msd.map(lambda l : [l.split(',')[1]]).take(40)
    #y = msd.map(lambda l : l.split(',')[2]).collect()

    x = msd.map(lambda l : l.split(',')).map(lambda l : (l[1], l[2]))
    print x.take(5)

    f1 = np.asarray(msd.map(lambda l : l.split(',')).map(lambda l : (float(l[1]), float(l[2]))).take(40))
    print type(f1)
    print f1

    plt.imshow(f1, cmap='gray')
    plt.show()

    #sns.set()
    #f1 = np.random.rand(10, 12)
    #print f1
    #sns.heatmap(f1)

    '''
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    '''

    #plt.imshow(x)
    #plt.show()

    #plt.imshow(f1, cmap='gray')
    #plt.show()

    '''
    labeled_points_df = sqlContext.createDataFrame(labeled_points)
    >> > data_values = (labeled_points_df >> >.rdd >> >.map(
lambda lp: [lp.features[1]]) >> >.take(40)) >> > len(data_values)
    40
    '''

    #data = rdd.map(lambda l : l.split(',')).map(lambda fields : (fields[1], fields[2])).collect()
    #sns.set()
    #ax = sns.heatmap(data)





