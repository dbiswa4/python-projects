import math
import re
import sys
import matplotlib.pyplot as plt
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import FloatType
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf


#Create Spark Context
sparkConf = SparkConf().setAppName("Onramp Spark 2 - Assignment 3")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

#Part 1
'''
Read each file and create RDD consisting of lines. First column of Google.csv is  URLs and first column of Amazon.csv is alphanumeric
strings. We call them ID to simplify and want to parse ID for each row. Load the data into RDD so that ID is the key and other
information are included in value.
'''
googlefile = sys.argv[1]
amazonfile = sys.argv[2]

google = sc.textFile(googlefile)
amazon = sc.textFile(amazonfile)

#Input data contains header which needs to be removed for further processing
google_wo_hdr = google.filter(lambda x: "id" not in x)
print 'Google input : ', google_wo_hdr.take(4)
amazon_wo_hdr = amazon.filter(lambda x: "id" not in x)
print 'Amazon input : ', amazon_wo_hdr.take(4)


#Part 2
'''
Bag-of-words is conceptually simple approach to text analysis. Here we treat each document as unordered collection of words.
We will construct some components for bag-of-words analysis on description field of datasets.
 (a) Implement a function that takes a string and returns non-empty tokens by splitting using regular expressions.
(b) Stopwords are common (English) words that do not contribute much to the content or meaning of a document
(e.g., "the", "a", "is", "to", etc.). Stopwords add noise to bag-of-words comparisons, so they are usually excluded. Using the included file "stopwords.txt", implement a function, an improved tokenizer that does not emit stopwords.
(c) Tokenize both Amazon and Google datasets. Here for each instance, tokenize the values corresponding to keys.
'''
def convert_to_pair_rdd(line):
    fields = line.split(',')
    return [fields[0], [fields[1], fields[2], fields[3], fields[4]]]

google_pair_rdd = google_wo_hdr.map(convert_to_pair_rdd)
print 'Google sample data :', google_pair_rdd.count()
amazon_pair_rdd = amazon_wo_hdr.map(convert_to_pair_rdd)
print 'Amazon sample data :', amazon_pair_rdd.count()


#Stop word removal
def initial_text_processing(tokens, stopwords):
    title,description,manufacturer,price = tokens
    words = re.sub(r'[^A-Za-z0-9 ]','',description).lower().strip().split()
    return [w for w in words if not w in stopwords and w !='']

stopwords_file = sys.argv[3]
stopwords = sc.textFile(stopwords_file)
stopwordslist = stopwords.collect()
googleclean = google_pair_rdd.mapValues(lambda x: initial_text_processing(x, stopwordslist))
amazonclean = amazon_pair_rdd.mapValues(lambda x: initial_text_processing(x, stopwordslist))


#Part 4
'''
Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and value is associated tokens from two datasets combined.
'''
corpus = googleclean.union(amazonclean).reduceByKey(lambda x, y: x + y)

print 'Total records in Corpus    : ', corpus.count()
print 'Sample records from Corpus : \n', corpus.take(5)


'''
Google sample data : 2252
Amazon sample data : 678
Total records in Corpus    :  2930
'''


#Part 3
'''
Write a function that takes a list of tokens and returns a dictionary mapping tokens to term frequency.
'''

def get_tf(document):
    freq = {}
    for term in document:
        i = term
        freq[i] = 1.0 if False else freq.get(i, 0) + 1.0
    termfreq = {}
    for key in freq.keys():
        termfreq[key] = freq[key]/len(document)
    return termfreq

tokens_tf_map = corpus.mapValues(lambda x: get_tf(x))

print 'Token and Term Frequency with document Id : ', tokens_tf_map.take(2)



'''
Token and Term Frequency with document Id :
[(u'"http://www.google.com/base/feeds/snippets/13707174407657102090"', {u'24': 0.07692307692307693, u'control': 0.07692307692307693, u'management': 0.07692307692307693, u'ibm': 0.07692307692307693, u'virtualcenter': 0.07692307692307693, u'ships': 0.07692307692307693, u'virtual': 0.07692307692307693, u'hours': 0.07692307692307693, u'infrastructure': 0.07692307692307693, u'take': 0.07692307692307693, u'usually': 0.07692307692307693, u'4817n24': 0.07692307692307693, u'software': 0.07692307692307693}), (u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.045454545454545456, u'features': 0.045454545454545456, u'selfexpression': 0.045454545454545456, u'easytouse': 0.045454545454545456, u'overview': 0.045454545454545456, u'printmaster': 0.045454545454545456, u'expect': 0.045454545454545456, u'upgrades': 0.045454545454545456, u'find': 0.045454545454545456, u'personal': 0.045454545454545456, u'make': 0.045454545454545456, u'creative': 0.045454545454545456, u'program': 0.045454545454545456, u'passion': 0.045454545454545456, u'new': 0.045454545454545456, u'endless': 0.045454545454545456, u'sophisticated': 0.045454545454545456, u'got': 0.045454545454545456, u'inspiration': 0.045454545454545456, u'great': 0.045454545454545456, u'brilliant': 0.045454545454545456, u'plus': 0.045454545454545456})]
'''



#Part 5
'''
Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value. Plot a histogram of IDF values.
'''

doc_count = corpus.count()
def get_idf(tokendf):
    return math.log((float(doc_count + 1) / float(tokendf + 1)))

def df(corpus):
    token_doc_map = corpus.flatMap(lambda x: [(token, x[0]) for token in x[1]])
    token_doc_map_df = sqlContext.createDataFrame(token_doc_map, ['token', 'doc_id'])
    token_doc_grp = token_doc_map_df.groupBy(token_doc_map_df.token, token_doc_map_df.doc_id).count()
    doc_freq = token_doc_grp.groupBy(token_doc_grp.token).count().withColumnRenamed('count', 'df')
    token_doc_map_df.show(10, truncate=False)
    token_doc_grp.show(10, truncate=False)
    doc_freq.show(10, truncate=False)

    return doc_freq

corpus_df = df(corpus)

idf_udf = udf(get_idf, FloatType())
idf_df = corpus_df.withColumn("idf", idf_udf("df"))
token_idf = idf_df.select(idf_df.token, idf_df.idf)

print 'Token with IDF     : ', token_idf.show(15, truncate=False)

'''
Token with IDF     :
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
|commander     |6.5968046|
|organizational|6.037189 |
|diskeeper     |6.373661 |
|chickens      |6.8844867|
|active        |6.5968046|
|taxefficient  |7.289952 |
|preview       |5.9036574|
+--------------+---------+
'''

#Histogram of IDF
print 'Plot Histogram for IDF'
token_idfs = token_idf.rdd.map(lambda x : x[1]).collect()
fig = plt.figure(figsize=(10, 5))
plt.hist(token_idfs, 50, log=True)
plt.show()


#Part 6
'''
Write a function which does the following on entire corpus:
(a) Calculate token frequencies for tokens
(b) Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight
'''

def get_tf_idf(tokens, idf):
    tf = get_tf(tokens)
    token_tfidf = {}
    for token in tf.keys():
        token_tfidf[token] = tf[token] * idf[token]
    return token_tfidf

token_idf_mappping = token_idf.rdd.collectAsMap()
tfidf = corpus.map(lambda x: (x[0], get_tf_idf(x[1], token_idf_mappping)))

print 'Token and TF-IDF : ', tfidf.take(2)

'''
Token and TF-IDF :
[(u'"http://www.google.com/base/feeds/snippets/13707174407657102090"', {u'24': 0.18663622782780576, u'control': 0.28511021687434274, u'management': 0.30867745326115537, u'ibm': 0.35776112629817086, u'virtualcenter': 0.46439915436964774, u'ships': 0.18545730297382063, u'virtual': 0.32657150121835565, u'hours': 0.17847218880286583, u'infrastructure': 0.445067258981558, u'take': 0.2770055624154898, u'usually': 0.1863398001744197, u'4817n24': 0.5607655231769269, u'software': 0.10464328068953295}), (u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.19519179517572577, u'features': 0.10890228098089046, u'selfexpression': 0.2744176821275191, u'easytouse': 0.17107233134183017, u'overview': 0.20826825228604404, u'printmaster': 0.25387289307334204, u'expect': 0.2582051753997803, u'upgrades': 0.25387289307334204, u'find': 0.1869044520638206, u'personal': 0.17601022937081076, u'make': 0.14839091084220193, u'creative': 0.17174574461850253, u'program': 0.1583313291723078, u'passion': 0.2744176821275191, u'new': 0.11046087741851807, u'endless': 0.2397749423980713, u'sophisticated': 0.23408571156588467, u'got': 0.21477283130992544, u'inspiration': 0.2582051753997803, u'great': 0.16661867228421298, u'brilliant': 0.2744176821275191, u'plus': 0.16784725405953146})]
'''









