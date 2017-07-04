import os
import re
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np


def preProcessFile():
  train_data_pos=open(file_path_pos_train)
  train_data_neg=open(file_path_neg_train)
  test_data=open(file_path_test_data)
  inputFiles = [train_data_pos,train_data_neg,test_data]
  write_to_csv=open(processedFilePathTrainData,"w")
  write_to_test_csv=open(processedFilePathTestData,"w")
  badChar="[,!.?#@=\n]" #list of bad characters to be removed from the SMS
  for file in inputFiles:
      count = 1
      for line in file:
          line=line.lower().replace("\t"," ")# First convert each word to lower case , then replace all tab space with single back space
          line=re.sub(badChar,"",line) # using regular expression remove all bad character from the SMS
          arr=line.split(" ")# split the line using space and put all the words into a list
          if file == train_data_pos:
              label="Positive"
          elif file == train_data_neg:
              label = "Negative"
          elif count < 2990:
              label = "Positive"
          else:
              label = "Negative"
          words= " ".join(word for word in arr[1:len(arr)]) # rest of the words in the list are joined back to form the original sentence
          toWrite=label+","+words # line to be written: class label, SMS
          if(file==train_data_neg or file==train_data_pos):
              write_to_csv.write(toWrite)
              write_to_csv.write("\n")
          else:
              write_to_test_csv.write(toWrite)
              write_to_test_csv.write("\n")
          count += 1
  file.close()
  write_to_csv.close()

def getDataAndLabel(fileName):
  file = open(fileName)# read the processed file
  label=[]
  data=[]
  for line in file:
      arr=line.replace("\n","").split(",") #split with comma
      label.append(arr[0])#first element is class label
      data.append(arr[1].replace("\n",""))#second element is SMS
  return data,label

def calBaseLine(data): # calculate baseline : it is percentage of records belonging to majority class
  classValues=np.unique(data) # from target values find out unique classes
  highest=0
  baseClass=""
  for label in classValues: # iterate over these classes to find number of records belonging to that class
      count=[i for i in data if i==label ] # create a list containing only label either ham or spam
      count=len(count) #find how many of them are  ham or spam
      if count>highest:
          highest=count
          baseClass=label
  print("Base Class :",baseClass)
  print("base Line :",(float(highest)/len(data))*100)

relativePath=os.getcwd()
file_path_pos_train= relativePath + "/Resources/TrainingDataPositive.txt"
file_path_neg_train= relativePath + "/Resources/TrainingDataNegative.txt"
file_path_test_data= relativePath + "/Resources/testSet.txt"
processedFilePathTrainData=relativePath+"/Resources/processedTrain.csv"
processedFilePathTestData=relativePath+"/Resources/processedTest.csv"
preProcessFile() #process the file
data,label=getDataAndLabel(processedFilePathTrainData) #get the data and label
dataTrain, dataTest, labelTrain, labelTest = train_test_split( data, label, test_size=0.33, random_state=42) #split the data and label into training set and test set . 2/3 is for training and 1/3 for testing
print("\nTraining Model\n")
count_vectorizer = CountVectorizer() # instance of count vectorize
train_count = count_vectorizer.fit_transform(dataTrain) # create a numerical feature vector
tfidf_transformer = TfidfTransformer() # calculate term frequency
train_tfidf = tfidf_transformer.fit_transform(train_count) #calculate Term Frequency times Inverse Document Frequency
model=MultinomialNB(fit_prior=True) # create an instance of multinomial Naive Bayes
model.fit(train_tfidf, labelTrain)# train the model
test_counts = count_vectorizer.transform(dataTest)
test_tfidf = tfidf_transformer.transform(test_counts)#create Term Frequency times Inverse Document Frequency for test data
predicted_labels = model.predict(test_tfidf)#predict the test data by using TFID
print("Accuracy on Test Data split :", np.mean(predicted_labels == labelTest) * 100) #calculte accuracy
calBaseLine(labelTest)

#Running the model on the Test Data provided
test_data,test_label=getDataAndLabel(processedFilePathTestData) #get the data and label
TestCount = count_vectorizer.transform(test_data)
TestTfidf = tfidf_transformer.transform(TestCount)
predtestdataLabel = model.predict(TestTfidf)
print("\nPredicting Labels of Test Data\n")
print("Accuracy on Test Data split :",np.mean(predtestdataLabel==test_label)*100) #calculte accuracy
calBaseLine(test_label)
