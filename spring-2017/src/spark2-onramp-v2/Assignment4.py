import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier


#Parse command like args

if len(sys.argv) < 2:
    print 'Not enough arguments passed'
    exit(1)
input_file_name = sys.argv[1]

#Start of the script
sparkConf = SparkConf().setAppName("Onramp Spark 2 - Assignment 4")
sc = SparkContext(conf=sparkConf)
sqlContext = SQLContext(sc)


#Perform transformation
def transform(line):
    tokens = line.split(',')
    if tokens[0].upper() == "ID" or tokens[0] == "" or tokens[0] is None:
        return
    pay_delay_avg = (float(tokens[6]) + float(tokens[7]) + float(tokens[8]) + float(tokens[9]) + float(tokens[10]) + float(tokens[11])) / 6.0
    bill_amt_avg = (float(tokens[12]) + float(tokens[13]) + float(tokens[14]) + float(tokens[15]) + float(tokens[16]) + float(tokens[17])) / 6.0
    payment_made_avg = (float(tokens[18]) + float(tokens[19]) + float(tokens[20]) + float(tokens[21]) + float(tokens[22]) + float(tokens[23])) / 6.0

    return (tokens[0], float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5]), float(pay_delay_avg), abs(float(bill_amt_avg)), float(payment_made_avg), float(tokens[24]))

input_data = sc.textFile(input_file_name).filter(lambda l: "X1" not in l and "ID" not in l)
credit_data = input_data.map(transform)
credit_data_lp = credit_data.map(lambda l: (l[9], Vectors.dense(l[2:9])))
credit_df = sqlContext.createDataFrame(credit_data_lp, ["label", "features"])

stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")

'''
Build and Evaluate model from transfomred data
'''

#Create a model evaluator
weighted_precision = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="weightedPrecision")
weighted_recall = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="weightedRecall")

# Train and Test set
(train_data, test_data) = credit_df.randomSplit([0.7, 0.3])
si_model_training = stringIndexer.fit(train_data)
train_data_indexed = si_model_training.transform(train_data)
si_model_test = stringIndexer.fit(train_data)
test_data_indexed = si_model_test.transform(test_data)


# Random Forest model
random_forest_class = RandomForestClassifier(numTrees=5, maxDepth=4, labelCol="indexed", seed=42)
random_forest_model = random_forest_class.fit(train_data_indexed)

# Prediction using Random Forest
predicted_result = random_forest_model.transform(test_data_indexed)
combined_result = predicted_result.select("label", "features", "prediction")
combined_result.show(80, False)
print 'Random Forest :: Weighted Pricision of model : ', weighted_precision.evaluate(predicted_result)
print 'Random Forest :: Weighted Recall of model    : ', weighted_recall.evaluate(predicted_result)

'''
 Random Forest :: Weighted Pricision of model : 0.776882205236
 Random Forest :: Weighted Recall of model    : 0.799712452997
'''


# Decision Tree model
decision_tree_class = DecisionTreeClassifier(maxDepth=4, labelCol="indexed")
decision_tree_model = decision_tree_class.fit(train_data_indexed)
# Prediction using Decision Tree
predicted_result = decision_tree_model.transform(test_data_indexed)
combined_result = predicted_result.select("label", "features", "prediction")
combined_result.show(80, False)
print 'Decision Tree :: Weighted Pricision of model : ', weighted_precision.evaluate(predicted_result)
print 'Decision Tree :: Weighted Recall of model    : ', weighted_recall.evaluate(predicted_result)

'''
Decision Tree :: Weighted Pricision of model :  0.778192687245
Decision Tree :: Weighted Recall of model    :  0.800597213006
'''