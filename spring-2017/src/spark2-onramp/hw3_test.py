from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.ml.feature import IDF as MLIDF
from pyspark.sql.types import DoubleType
from pyspark.mllib.linalg import (Vector, Vectors, DenseVector, SparseVector, _convert_to_vector)
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, MapType, FloatType
import math


spark = SparkSession.builder.config("spark.sql.warehouse.dir", "spark-workspace").appName("TF IDF Demo").getOrCreate()
sc = spark.sparkContext

documents = sqlContext.createDataFrame([     (0, "hello spark spark", "data1"),     (1, "this this is is example", "data2"),     (2, "spark is fast","data3"),     (3, "hello world","data4")], ["doc_id", "doc_text", "another"])

documents.printSchema()

df = (documents   .rdd   .map(lambda x : (x.doc_id,x.doc_text.split(" ")))   .toDF()   .withColumnRenamed("_1","doc_id")   .withColumnRenamed("_2","features"))


htf = MLHashingTF(inputCol="features", outputCol="tf")
tf = htf.transform(df)
tf.show(truncate=False)


idf = MLIDF(inputCol="tf", outputCol="idf")
tfidf = idf.fit(tf).transform(tf)
tfidf.show(truncate=False)



res = tfidf.rdd.map(lambda x : (x.doc_id,x.features,x.tf,x.idf,(None if x.idf is None else x.idf.values.sum())))
for r in res.take(10):
  print r


#################
#Experiment 1
#numFeatures value set
documents = sqlContext.createDataFrame([(0, "hello spark spark", "data1"),(1, "hello spark spark", "data2"),(2, "spark spark is fast","data3"),(3, "hello world","data4")], ["doc_id", "doc_text", "another"])
df = (documents.rdd.map(lambda x : (x.doc_id,x.doc_text.split(" "))).toDF().withColumnRenamed("_1","doc_id").withColumnRenamed("_2","features"))

htf = MLHashingTF(inputCol="features", outputCol="tf", numFeatures=32)
tf = htf.transform(df)
tf.show(truncate=False)

idf = MLIDF(inputCol="tf", outputCol="idf")
tfidf = idf.fit(tf).transform(tf)
tfidf.show(truncate=False)

##!!!
###tf method :  return hash(term) % self.numFeatures
#Why the hell hash('spark') is 1?
#If I do it in python the hash is 26
#Lokks like it's because Spark is using murmur hash : MurmurHash3_x86_32
numFeatures = 32
term = 'spark'
print 'hash(term) % numFeatures : ', hash(term) % numFeatures


'''
+------+------------------------+----------------------------+
|doc_id|features                |tf                          |
+------+------------------------+----------------------------+
|0     |[hello, spark, spark]   |(32,[1,8],[2.0,1.0])        |
|1     |[hello, spark, spark]   |(32,[1,8],[2.0,1.0])        |
|2     |[spark, spark, is, fast]|(32,[1,17,31],[2.0,1.0,1.0])|
|3     |[hello, world]          |(32,[8,30],[1.0,1.0])       |
+------+------------------------+----------------------------+

+------+------------------------+----------------------------+--------------------------------------------------------------------------+
|doc_id|features                |tf                          |idf                                                                       |
+------+------------------------+----------------------------+--------------------------------------------------------------------------+
|0     |[hello, spark, spark]   |(32,[1,8],[2.0,1.0])        |(32,[1,8],[0.44628710262841953,0.22314355131420976])                      |
|1     |[hello, spark, spark]   |(32,[1,8],[2.0,1.0])        |(32,[1,8],[0.44628710262841953,0.22314355131420976])                      |
|2     |[spark, spark, is, fast]|(32,[1,17,31],[2.0,1.0,1.0])|(32,[1,17,31],[0.44628710262841953,0.9162907318741551,0.9162907318741551])|
|3     |[hello, world]          |(32,[8,30],[1.0,1.0])       |(32,[8,30],[0.22314355131420976,0.9162907318741551])                      |
+------+------------------------+----------------------------+--------------------------------------------------------------------------+
'''


