from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
import re
import math

#Create Spark Context
sparkConf = SparkConf().setAppName("Spark 2 practice - asssignment 2")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

'''
Part 1
Read each file and create RDD consisting of lines. First column of Google.csv is  URLs and first column of Amazon.csv
is alphanumeric strings. We call them ID to simplify and want to parse ID for each row. Load the data into RDD so that ID
is the key and other information are included in value.
'''
google_data = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/Google.csv')
amazon_data = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/Amazon.csv')

#Remove header
google_data_no_hdr = google_data.filter(lambda x: "id" not in x)
amazon_data_no_hdr = amazon_data.filter(lambda x: "id" not in x)

print 'Google Data : \n', google_data_no_hdr.take(2)
print 'Amazon Data : \n', amazon_data_no_hdr.take(2)


#Part 2
#Data formatting and transformation

'''
Part 2
Bag-of-words is conceptually simple approach to text analysis. Here we treat each document as unordered collection of words.
We will construct some components for bag-of-words analysis on description field of datasets.
 (a) Implement a function that takes a string and returns non-empty tokens by splitting using regular expressions.
(b) Stopwords are common (English) words that do not contribute much to the content or meaning of a document (e.g., "the", "a", "is", "to", etc.). Stopwords add noise to bag-of-words comparisons, so they are usually excluded. Using the included file "stopwords.txt",
implement a function, an improved tokenizer that does not emit stopwords.
(c) Tokenize both Amazon and Google datasets. Here for each instance, tokenize the values corresponding to keys.
'''
def create_pair_rdd(line):
    fields = line.split(',')
    return [fields[0], [fields[1], fields[2], fields[3], fields[4]]]

google_kv = google_data_no_hdr.map(create_pair_rdd)
amazon_kv = amazon_data_no_hdr.map(create_pair_rdd)

print 'Google kv:\n', google_kv.take(3)
print 'Amazon kv\n:', amazon_kv.take(3)
print 'Google kv count :', google_kv.count()
print 'Amazon kv count :', amazon_kv.count()

#Stop word removal and tokenization
def text_processing(tokens, stopwords):
    title,desc,manufac,price = tokens
    words = re.sub(r'[^A-Za-z0-9 ]','',desc).lower().strip().split()
    return [w for w in words if not w in stopwords and w !='']

stopwords = sc.textFile('/Users/dbiswas/Documents/MS/Spring2017/ramp-spark2/files/stopwords.txt')
stopwords_list = stopwords.collect()
google_clean = google_kv.mapValues(lambda x: text_processing(x, stopwords_list))
amazon_clean = amazon_kv.mapValues(lambda x: text_processing(x, stopwords_list))


'''
Part 4
Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and value is
associated tokens from two datasets combined.
'''

corpus = google_clean.union(amazon_clean).reduceByKey(lambda x, y: x+y)

print '\nGoogle cleaned records : ', google_clean.count()
print 'Amazon cleaned records   : ', amazon_clean.count()
print 'Corpus count             : ', corpus.count()
print 'Google tokenized:\n', google_clean.take(5)
print 'Amazon tokenized():\n', amazon_clean.take(5)
print 'Corpus : \n', corpus.take(5)



'''
Part 3
Write a function that takes a list of tokens and returns a dictionary mapping tokens to term frequency.
'''
def tf(document):
    freq = dict()
    for term in document:
        i = term
        freq[i] = 1.0 if False else freq.get(i, 0) + 1.0
    termfreq = {}
    for key in freq.keys():
        termfreq[key] = freq[key]/len(document)
    return termfreq

tokens_tf_map = corpus.mapValues(lambda x: tf(x))
print 'Term Frequency : ', tokens_tf_map.take(2)

'''
Term Frequency :  [(u'"http://www.google.com/base/feeds/snippets/13707174407657102090"', {u'24': 0.07692307692307693, u'control': 0.07692307692307693, u'management': 0.07692307692307693, u'ibm': 0.07692307692307693, u'virtualcenter': 0.07692307692307693, u'ships': 0.07692307692307693, u'virtual': 0.07692307692307693, u'hours': 0.07692307692307693, u'infrastructure': 0.07692307692307693, u'take': 0.07692307692307693, u'usually': 0.07692307692307693, u'4817n24': 0.07692307692307693, u'software': 0.07692307692307693}), (u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.045454545454545456, u'features': 0.045454545454545456, u'selfexpression': 0.045454545454545456, u'easytouse': 0.045454545454545456, u'overview': 0.045454545454545456, u'printmaster': 0.045454545454545456, u'expect': 0.045454545454545456, u'upgrades': 0.045454545454545456, u'find': 0.045454545454545456, u'personal': 0.045454545454545456, u'make': 0.045454545454545456, u'creative': 0.045454545454545456, u'program': 0.045454545454545456, u'passion': 0.045454545454545456, u'new': 0.045454545454545456, u'endless': 0.045454545454545456, u'sophisticated': 0.045454545454545456, u'got': 0.045454545454545456, u'inspiration': 0.045454545454545456, u'great': 0.045454545454545456, u'brilliant': 0.045454545454545456, u'plus': 0.045454545454545456})]
'''


'''
Part 5
Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value.
Plot a histogram of IDF values.
Note: You can use MLLIB for TF and IDF functions
'''

N = corpus.count()
def idf(tokendf):
    return math.log((float(N + 1) / float(tokendf + 1)))

def df(corpus):
    docid_token = corpus.flatMap(lambda x: [(token, x[0]) for token in x[1]])
    docid_token_df = sqlContext.createDataFrame(docid_token, ['token', 'doc_id'])
    token_doc_grp = docid_token_df.groupBy(docid_token_df.token, docid_token_df.doc_id).count()
    doc_freq = token_doc_grp.groupBy(token_doc_grp.token).count().withColumnRenamed('count', 'df')
    docid_token_df.show(10, truncate=False)
    token_doc_grp.show(10, truncate=False)
    doc_freq.show(10, truncate=False)
    return doc_freq

corpusdf = df(corpus)

calculate_idf = udf(idf, FloatType())
idf_df = corpusdf.withColumn("idf", calculate_idf("df"))
token_idf = idf_df.select(idf_df.token, idf_df.idf)

print 'Tokens and IDF     : ', token_idf.show(10, truncate=False)

#Plotting
print 'Plot idf histogram'
token_idf_values = token_idf.rdd.map(lambda x : x[1]).collect()
fig = plt.figure(figsize=(8, 3))
plt.hist(token_idf_values, 50, log=True)
plt.show()

'''
Part 6
Write a function which does the following on entire corpus:
(a) Calculate token frequencies for tokens
(b) Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight
Note: TF  x IDF is also known as tf-idf in NLP terminology.
'''
def tf_idf(tokens, idf):
    token_tf = tf(tokens)
    tfidf = dict()
    for token in token_tf.keys():
        tfidf[token] = token_tf[token] * idf[token]
    return tfidf


idf_map = token_idf.rdd.collectAsMap()
tfidf = corpus.map(lambda x: (x[0], tf_idf(x[1], idf_map)))

print 'TF-IDF : ', tfidf.take(2)

'''
TF-IDF :  [(u'"http://www.google.com/base/feeds/snippets/13707174407657102090"', {u'24': 0.18663622782780576, u'control': 0.28511021687434274, u'management': 0.30867745326115537, u'ibm': 0.35776112629817086, u'virtualcenter': 0.46439915436964774, u'ships': 0.18545730297382063, u'virtual': 0.32657150121835565, u'hours': 0.17847218880286583, u'infrastructure': 0.445067258981558, u'take': 0.2770055624154898, u'usually': 0.1863398001744197, u'4817n24': 0.5607655231769269, u'software': 0.10464328068953295}), (u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.19519179517572577, u'features': 0.10890228098089046, u'selfexpression': 0.2744176821275191, u'easytouse': 0.17107233134183017, u'overview': 0.20826825228604404, u'printmaster': 0.25387289307334204, u'expect': 0.2582051753997803, u'upgrades': 0.25387289307334204, u'find': 0.1869044520638206, u'personal': 0.17601022937081076, u'make': 0.14839091084220193, u'creative': 0.17174574461850253, u'program': 0.1583313291723078, u'passion': 0.2744176821275191, u'new': 0.11046087741851807, u'endless': 0.2397749423980713, u'sophisticated': 0.23408571156588467, u'got': 0.21477283130992544, u'inspiration': 0.2582051753997803, u'great': 0.16661867228421298, u'brilliant': 0.2744176821275191, u'plus': 0.16784725405953146})]
'''










