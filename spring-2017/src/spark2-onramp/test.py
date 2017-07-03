from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


if __name__ == '__main__':
    sparkConf = SparkConf().setAppName("Spark Test")
    sc = SparkContext(conf=sparkConf)
    sqlContext = SQLContext(sc)

    rdd = sc.textFile('/Users/dbiswas/Documents/SourceCode/python/python-projects/spring-2017/src/spark2-onramp/scatter.txt')
    print rdd.first()

    data = rdd.map(lambda l: l.split(',')).map(lambda l: (l[0], l[1]))
    print data.take(3)

    actual = np.asarray(data.map(lambda l : l[0]).collect())
    predicted = np.asarray(data.map(lambda l: l[1]).collect())

    plt.scatter(predicted, actual, alpha=0.5)
    plt.title('Year : Predicted vs Actual')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()






