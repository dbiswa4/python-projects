from pyspark.sql import SparkSession
from pyspark.sql import Row
import re
from pyspark.ml.feature import HashingTF, IDF
from pyspark.sql.types import *
from pyspark.mllib.linalg import (Vector, Vectors, DenseVector, SparseVector, _convert_to_vector)
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType, FloatType, DoubleType
import math

spark = SparkSession.builder.config("spark.sql.warehouse.dir", "spark-workspace").appName("DS OnRamp Spark 2 - hw3").getOrCreate()
sc = spark.sparkContext

#Part 1
#Data loading
google_data = sc.textFile('/FileStore/tables/zokdbgdq1490943916146/Google.csv')
amazon_data = sc.textFile('/FileStore/tables/d4div6y91490943981098/Amazon.csv')

#Remove header from Google Data
google_data_no_hdr = google_data.filter(lambda x: "id" not in x)
print 'type(google_data_no_hdr) : ', type(google_data_no_hdr)
google_data_no_hdr.first()

#Remove header from Amazon data
amazon_data_no_hdr = amazon_data.filter(lambda x: "id" not in x)
print 'type(amazon_data_no_hdr) : ', type(amazon_data_no_hdr)
amazon_data_no_hdr.take(2)


#Part 2
#Data formatting and transformation
def make_data_kv2(line):
    fields = line.split(',')
    return [fields[0], [[fields[1], fields[2], fields[3], fields[4]]]]

google_kv2 = google_data_no_hdr.map(make_data_kv2)
print '\nGoogle kv:', google_kv2.take(3)

amazon_kv2 = google_data_no_hdr.map(make_data_kv2)
print '\nAmazon kv:', amazon_kv2.take(3)

x = google_kv2.reduceByKey(lambda a, b: a + b)
print '\nreduceByKey:'
print x.take(3)

def preprocess_text_new(values, stopwords):
    for value in values:
        title,description,manufacturer,price = value
        words = re.sub(r'[^A-Za-z0-9 ]','',description).lower().strip().split()
        return [w for w in words if not w in stopwords and w !='']


stopwords = sc.textFile('/FileStore/tables/oz84gy521491024785846/stopwords.txt')
stopwords_list = stopwords.collect()

google_clean = google_kv2.mapValues(lambda a: preprocess_text_new(a, stopwords_list))
print '\nGoogle cleaned records : ', google_clean.count()
print '\nGoogle mapValues()', google_clean.take(3)

#Amazon data
amazon_clean = amazon_kv2.mapValues(lambda a: preprocess_text_new(a, stopwords_list))
print '\nAmazon cleaned records : ', amazon_clean.count()
print '\nAmazon mapValues()', amazon_clean.take(3)


#Part 3
#Write a function that takes a list of tokens and returns a dictionary mapping tokens to weights.
#Q. What is definition of 'weights'?



#Part 4
#Combine the datasets to create a corpus. Each element of the corpus is a <key, value> pair where key is ID and value is
# associated tokens from two datasets combined.
combined_data = google_clean.union(amazon_clean)
print '\ncombined_data record count : ', combined_data.count()
print '\ncombined_data:', combined_data.take(3)

corpus = google_clean.union(amazon_clean).reduceByKey(lambda a, b: a+b)
print '\ncorpus count : ', corpus.count()
print '\ncorpus:', combined_data.take(3)



#Part 5 - Final Solution
#Write an IDF function that return a pair RDD where key is each unique token and value is corresponding IDF value.
# Plot a histogram of IDF values.
token_doc_map = corpus.flatMap(lambda x: [(token, x[0]) for token in x[1]])
print '\ntoken_doc_map'
print token_doc_map.take(10)

token_doc_map_df = token_doc_map.toDF().withColumnRenamed("_1","token").withColumnRenamed("_2","doc_id")
token_doc_map_df.show(truncate=False)
#Group By token and doc id : gives Term Frequency
token_doc_grp = token_doc_map_df.groupBy(token_doc_map_df.token, token_doc_map_df.doc_id).count()
token_doc_grp.show(truncate=False)
#Group By token. Gives Document Frequency. ToDo:Can do it in one pass
doc_freq = token_doc_grp.groupBy(token_doc_grp.token).count().withColumnRenamed("count","df")
doc_freq.show(truncate=False)

total_doc = corpus.count()
def idf(tokendf):
    return math.log((float(total_doc+1)/float(tokendf+1)))

tokendfUDF = udf(idf, FloatType())
idfDF = doc_freq.withColumn("idf", tokendfUDF("df"))
idfDF.show(truncate=False)

#Token and IDF RDD
idfrdd = idfDF.select(idfDF.token, idfDF.idf).rdd.map(lambda x:(x.token, x.idf))
print '\n idfrdd : ', type(idfrdd)
print idfrdd.take(3)


#Plot Histogram of idf
#ToDo: "Can not generate buckets with non-number in RDD"
hist = idfDF.select(idfDF.idf).rdd.histogram(10)



#Part 6
#Write a function which does the following:
#(a) Calculate token frequencies for tokens
#(b) Create a Python dictionary where each token maps to the token's frequency times the token's IDF weight

#a. Token Frequencies : Total number of times a word/token appear in entire corpus
tokenfqDF = corpus.flatMap(lambda x : x[1]).map(lambda x: (x,1)).reduceByKey(lambda x, y: (x + y)).toDF().withColumnRenamed("_1","token").withColumnRenamed("_2","frequency")
tokenfqDF.show(truncate=False)

#b. Generate : token, token's frequency*token's IDF weight
token_stats = idfDF.join(tokenfqDF, idfDF.token == tokenfqDF.token).select(idfDF.token,idfDF.df,idfDF.idf, tokenfqDF.frequency)
token_stats.show(truncate=False)
freq_idf_df = token_stats.withColumn("freq_times_idf", token_stats['frequency'] * token_stats['idf'])
freq_idf_df.show(5)

freq_idf_dict = freq_idf_df.select(freq_idf_df.token,freq_idf_df.freq_times_idf).collect()






