from pyspark import SparkContext, SparkConf
import re
import sys

def normalizeWords(text):
    words = re.sub(r'[^A-Za-z0-9 ]','',text).lower().strip().split()
    return [w for w in words if len(w) > 3 and w !='']

if __name__ == '__main__':
    sparkConf = SparkConf().setMaster("local").setAppName("Word Count Better - Sorted")
    sc = SparkContext(conf=sparkConf)

    filename = sys.argv[1]
    input = sc.textFile(filename)
    words = input.flatMap(normalizeWords)
    word_counts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: (x + y))
    word_counts_sorted = word_counts.map(lambda (x, y): (y, x)).sortByKey(False).map(lambda (x,y): (y, x))

    print 'Wordcount : \n', word_counts_sorted.take(20)

    '''
    Wordcount :
    [(u'that', 389), (u'this', 300), (u'with', 277), (u'your', 242), (u'lord', 225), (u'what', 208), (u'king', 196), (u'have', 183), (u'will', 172), (u'queen', 120), (u'shall', 114), (u'good', 109), (u'come', 107), (u'hamlet', 107), (u'thou', 105), (u'they', 101), (u'more', 95), (u'from', 95), (u'well', 90), (u'most', 82)]
    '''

    results = word_counts_sorted.collect()

    for result in results:
        print result
        count = str(result[1])
        word = result[0].encode('ascii', 'ignore')
        if(word):
            print word + " : " + count