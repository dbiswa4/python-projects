from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
import re
import math
import sys

#Create Spark Context
sparkConf = SparkConf().setAppName("Spark 2 practice - Assignment 3")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)

'''
Part 1
Read each file and create RDD consisting of lines. First column of Google.csv is  URLs and first column of Amazon.csv
is alphanumeric strings. We call them ID to simplify and want to parse ID for each row. Load the data into RDD so that ID
is the key and other information are included in value.
'''
google_data_file = sys.argv[1]
amazon_data_file = sys.argv[2]
google_data = sc.textFile(google_data_file)
amazon_data = sc.textFile(amazon_data_file)

# google_data = sc.textFile('C:/Users/mail2/Spark/Assignment/Assignment3/Google.csv')
# amazon_data = sc.textFile('C:/Users/mail2/Spark/Assignment/Assignment3/Amazon.csv')

#Remove header
google_data_wo_header = google_data.filter(lambda x: "id" not in x)
amazon_data_wo_header = amazon_data.filter(lambda x: "id" not in x)

print '****************Google Data without header******************** \n', google_data_wo_header.take(2)
print '****************Amazon Data without header******************** \n', amazon_data_wo_header.take(2)



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

google_key_value = google_data_wo_header.map(create_pair_rdd)
print 'Google Key Values:\n', google_key_value.take(3)
print 'Google kv count :', google_key_value.count()

amazon_key_value = amazon_data_wo_header.map(create_pair_rdd)
print 'Amazon kv\n:', amazon_key_value.take(3)
print 'Amazon kv count :', amazon_key_value.count()

#Tokenizing and removing stop word
def text_processing(tokens, stopwords):
    title, desc, manufac, price = tokens
    words = re.sub(r'[^A-Za-z0-9 ]','',desc).lower().strip().split()
    return [w for w in words if not w in stopwords and w !='']


stopwordsfile = sys.argv[3]
stopwords = sc.textFile(stopwordsfile)


# stopwords = sc.textFile('C:/Users/mail2/Spark/Assignment/Assignment3/stopwords.txt')
stopwords_list = stopwords.collect()
google_clean = google_key_value.mapValues(lambda x: text_processing(x, stopwords_list))
amazon_clean = amazon_key_value.mapValues(lambda x: text_processing(x, stopwords_list))


'''
Part 4
Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and value is
associated tokens from two datasets combined.
'''

dataset_corpus = google_clean.union(amazon_clean).reduceByKey(lambda x, y: x + y)

print '\nGoogle cleaned records : ', google_clean.count()
print 'Amazon cleaned records   : ', amazon_clean.count()
print 'Corpus count             : ', dataset_corpus.count()
print 'Google tokenized:\n', google_clean.take(5)
print 'Amazon tokenized():\n', amazon_clean.take(5)
print 'Corpus : \n', dataset_corpus.take(5)



'''
Part 3
Write a function that takes a list of tokens and returns a dictionary mapping tokens to term frequency.
'''
def term_freq_func(document):
    freq = dict()
    for term in document:
        i = term
        freq[i] = 1 if False else freq.get(i, 0) + 1
    termfreq = {}
    for key in freq.keys():
        termfreq[key] = freq[key]/len(document)
    return termfreq

tokens_tf_map = dataset_corpus.mapValues(lambda x: term_freq_func(x))
print 'Term Frequency : ', tokens_tf_map.take(5)

'''
Term Frequency :  [(u'"http://www.google.com/base/feeds/snippets/13707174407657102090"', {u'24': 0.07692307692307693, u'control': 0.07692307692307693, u'management': 0.07692307692307693, u'ibm': 0.07692307692307693, u'virtualcenter': 0.07692307692307693, u'ships': 0.07692307692307693, u'virtual': 0.07692307692307693, u'hours': 0.07692307692307693, u'infrastructure': 0.07692307692307693, u'take': 0.07692307692307693, u'usually': 0.07692307692307693, u'4817n24': 0.07692307692307693, u'software': 0.07692307692307693}), (u'"http://www.google.com/base/feeds/snippets/5558998293398547766"', {u'enjoy': 0.045454545454545456, u'features': 0.045454545454545456, u'selfexpression': 0.045454545454545456, u'easytouse': 0.045454545454545456, u'overview': 0.045454545454545456, u'printmaster': 0.045454545454545456, u'expect': 0.045454545454545456, u'upgrades': 0.045454545454545456, u'find': 0.045454545454545456, u'personal': 0.045454545454545456, u'make': 0.045454545454545456, u'creative': 0.045454545454545456, u'program': 0.045454545454545456, u'passion': 0.045454545454545456, u'new': 0.045454545454545456, u'endless': 0.045454545454545456, u'sophisticated': 0.045454545454545456, u'got': 0.045454545454545456, u'inspiration': 0.045454545454545456, u'great': 0.045454545454545456, u'brilliant': 0.045454545454545456, u'plus': 0.045454545454545456})]
'''


'''
Part 5
Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value.
Plot a histogram of IDF values.
Note: You can use MLLIB for TF and IDF functions
'''

total = dataset_corpus.count()


def idf(tokendf):
    return math.log((total + 1.0)/(tokendf + 1))


def df(corpus):
    df_token = corpus.flatMap(lambda x: [(token, x[0]) for token in x[1]])
    df_token_df = sqlContext.createDataFrame(df_token, ['token', 'doc_id'])
    token_doc_grp = df_token_df.groupBy(df_token_df.token, df_token_df.doc_id).count()
    doc_freq = token_doc_grp.groupBy(token_doc_grp.token).count().withColumnRenamed('count', 'df')
    df_token_df.show(10, truncate=False)
    token_doc_grp.show(10, truncate=False)
    doc_freq.show(10, truncate=False)
    return doc_freq

corpus_data_freq = df(dataset_corpus)

calculate_idf = udf(idf, FloatType())
idf_df = corpus_data_freq.withColumn("idf", calculate_idf("df"))
token_idf = idf_df.select(idf_df.token, idf_df.idf)

print '*****************Tokens and IDF**********\n', token_idf.show(10, truncate=False)

#Plotting on Histogram
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
def term_freq_idf(tokens, idf):
    token_term_freq = term_freq_func(tokens)
    tfidf = dict()
    for token in token_term_freq.keys():
        tfidf[token] = token_term_freq[token] * idf[token]
    return tfidf


idf_map = token_idf.rdd.collectAsMap()
tfidf = dataset_corpus.map(lambda x: (x[0], term_freq_idf(x[1], idf_map)))

print 'TF-IDF : ', tfidf.take(5)
