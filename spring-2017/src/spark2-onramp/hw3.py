import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import re
from collections import Counter
from pyspark.mllib.feature import HashingTF
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import matplotlib.pyplot as plt
import math

#Create Spark Context
sparkConf = SparkConf().setAppName("DS OnRamp Spark 2 - hw3")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

#Part 1
#Data loading
google_data = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/Google.csv')
amazon_data = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/Amazon.csv')

#Remove header
google_data_no_hdr = google_data.filter(lambda x: "id" not in x)
amazon_data_no_hdr = amazon_data.filter(lambda x: "id" not in x)

print 'Google Data : \n', google_data_no_hdr.take(2)
print 'Amazon Data : \n', amazon_data_no_hdr.take(2)


#Part 2
#Data formatting and transformation
def make_data_kv(line):
    fields = line.split(',')
    return [fields[0], [fields[1], fields[2], fields[3], fields[4]]]

google_kv = google_data_no_hdr.map(make_data_kv)
amazon_kv = amazon_data_no_hdr.map(make_data_kv)

print 'Google kv:\n', google_kv.take(3)
print 'Amazon kv\n:', amazon_kv.take(3)
print 'Google kv count :', google_kv.count()
print 'Amazon kv count :', amazon_kv.count()

#Tokenize the description and remove stop words
def preprocess_text(fields, stopwords):
    title,description,manufacturer,price = fields
    words = re.sub(r'[^A-Za-z0-9 ]','',description).lower().strip().split()
    return [w for w in words if not w in stopwords and w !='']

stopwords = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/stopwords.txt')
stopwords_list = stopwords.collect()
google_clean = google_kv.mapValues(lambda x: preprocess_text(x, stopwords_list))
amazon_clean = amazon_kv.mapValues(lambda x: preprocess_text(x, stopwords_list))


#Part 4
#Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and
# value is associated tokens from two datasets combined.
corpus = google_clean.union(amazon_clean).reduceByKey(lambda a, b: a+b)

print '\nGoogle cleaned records : ', google_clean.count()
print 'Amazon cleaned records   : ', amazon_clean.count()
print 'Corpus count             : ', corpus.count()
print 'Google tokenized:\n', google_clean.take(3)
print 'Amazon tokenized():\n', amazon_clean.take(3)
print 'Corpus : \n', corpus.take(5)



#Part 3
#Write a function that takes a list of tokens and returns a dictionary mapping tokens to term frequency.
#Note: You can use MLLIB for TF and IDF functions

#Part3-Sol2
#Gives right result
def transform(document):
    freq = {}
    for term in document:
        i = term
        freq[i] = 1.0 if False else freq.get(i, 0) + 1.0
    newfreq = {}
    for key in freq.keys():
        newfreq[key] = freq[key]/len(document)
    return newfreq

tokens_tf = corpus.map(lambda x: transform(x[1]))
tokens_tf_map = corpus.mapValues(lambda x: transform(x))
print 'Part3-Sol2:'
print 'Token and Term Frequency : ', tokens_tf.take(5)
print 'Token and Term Frequency (mapping with doc id retained) : ', tokens_tf_map.take(5)

'''
[(u'"http://www.google.com/base/feeds/snippets/10225446033682010691"', {u'12421': 0.2, u'workshop': 0.1, u'adventure': 0.1, u'8': 0.1, u'1st3rd': 0.1, u'encore': 0.2, u'software': 0.2})]
'''

#Part3-Sol3
#It gives term count, not frequency
hashingTF = HashingTF()
tokens_tf_map = corpus.mapValues(lambda x: hashingTF.transform(x))
print 'Part3-Sol3:'
print 'Token and Term Frequency (mapping with doc id retained) : ', tokens_tf_map.take(5)

#Part 5
#Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value.
# Plot a histogram of IDF values.
#Note: You can use MLLIB for TF and IDF functions

total_doc = corpus.count()
def idf(tokendf):
    return math.log((float(total_doc+1)/float(tokendf+1)))

def df(corpus):
    token_doc_map = corpus.flatMap(lambda x: [(token, x[0]) for token in x[1]])
    token_doc_map_df = sqlContext.createDataFrame(token_doc_map, ['token', 'doc_id'])
    # Group By token and doc id : it gives in which all document a token appears
    token_doc_grp = token_doc_map_df.groupBy(token_doc_map_df.token, token_doc_map_df.doc_id).count()
    # Group By token. Gives Document Frequency. ToDo:Can do it in one pass
    doc_freq = token_doc_grp.groupBy(token_doc_grp.token).count().withColumnRenamed('count', 'df')

    print 'token_doc_map    : \n', token_doc_map.take(10)
    token_doc_map_df.show(10, truncate=False)
    token_doc_grp.show(10, truncate=False)
    doc_freq.show(10, truncate=False)

    return doc_freq

corpus_df = df(corpus)

tokendfUDF = udf(idf, FloatType())
idf_df = corpus_df.withColumn("idf", tokendfUDF("df"))
token_idf = idf_df.select(idf_df.token, idf_df.idf)


print 'Calculated df : \n', corpus_df.show(10, truncate=False)
print 'idf_df        : \n', idf_df.show(10, truncate=False)
print 'token_idf     : ', token_idf.show(10, truncate=False)

'''
idf_df :
+--------------+---+---------+
|token         |df |idf      |
+--------------+---+---------+
|adventure     |44 |4.1764364|
|check         |40 |4.269527 |
|extending     |2  |6.8844867|
|frogger       |1  |7.289952 |
|tile          |3  |6.5968046|
|sub           |13 |5.344042 |

token_idf:
+--------------+---------+
|token         |idf      |
+--------------+---------+
|adventure     |4.1764364|
|check         |4.269527 |
|extending     |6.8844867|
|frogger       |7.289952 |
|tile          |6.5968046|
|sub           |5.344042 |
|moving        |5.9036574|
|preferences   |6.5968046|
'''

#Plotting
print 'Plot idf histogram'
token_idf_values = token_idf.rdd.map(lambda x : x[1]).collect()
fig = plt.figure(figsize=(8, 3))
plt.hist(token_idf_values, 50, log=True)
#plt.savefig('idf_hist.png')
plt.show()


#Part 6
#Write a function which does the following on entire corpus:
#(a) Calculate token frequencies for tokens
#(b) Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight

def tf_idf(tokens, idf):
    tf = transform(tokens)
    token_tfidf = {}
    for token in tf.keys():
        token_tfidf[token] = tf[token] * idf[token]
    return token_tfidf


token_idf_map = token_idf.rdd.collectAsMap()
tfidf = corpus.map(lambda x: (x[0], tf_idf(x[1], token_idf_map)))

print 'tfidf : ', tfidf.take(5)

'''
tfidf :
[(u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.19519179517572577, u'features': 0.10890228098089046, u'selfexpression': 0.2744176821275191, u'easytouse': 0.17107233134183017, u'overview': 0.20826825228604404, u'printmaster': 0.25387289307334204, u'expect': 0.2582051753997803, u'upgrades': 0.25387289307334204, u'find': 0.1869044520638206, u'personal': 0.17601022937081076, u'make': 0.14839091084220193, u'creative': 0.17174574461850253, u'program': 0.1583313291723078, u'passion': 0.2744176821275191, u'new': 0.11046087741851807, u'endless': 0.2397749423980713, u'sophisticated': 0.23408571156588467, u'got': 0.21477283130992544, u'inspiration': 0.2582051753997803, u'great': 0.16661867228421298, u'brilliant': 0.2744176821275191, u'plus': 0.16784725405953146})]
'''










