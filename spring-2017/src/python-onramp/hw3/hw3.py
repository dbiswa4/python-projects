"""
1. You have been given a barGraph.csv file.
Using the data of this file you have to draw a bar graph showing all 8 emotions corresponding to each business.
"""

import pandas
import os
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/barGraph.csv"
data = pandas.read_csv(dataFilePath)

index = np.arange(len(data.enjoyment))
barWidth = 0.1
opacity = 0.4

columns = data.columns

count = 0
for column in columns.tolist()[1:]:
    plt.bar(index+count*barWidth,data[column].tolist(),barWidth,label=column)
    count+=1
    if count==int(len(columns.tolist())/2):
        plt.xticks(index+count*barWidth,data['Business'].tolist())

plt.legend()
loc, labels = plt.xticks()
plt.setp(labels)
plt.show()

"""
2. Using the data present in barGraph.csv file generate pie-chart showing percentage of emotions for each business.
"""

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/barGraph.csv"
data = pandas.read_csv(dataFilePath)

labels = data.columns[1:].tolist()
count = 1
columns, b = plt.subplots(3, 2)
for i in range(3):
    for j in range(2):
        b[i,j].pie(data.loc[i+j][1:].tolist(),autopct='%1.1f%%', radius=0.8)
        b[i,j].set_title(data.loc[i+j][0])
        b[i,j].legend(labels,loc="upper left", fontsize=6)
plt.show()


"""
3. Generate a word cloud of your favorite news article or story or anything.
This word cloud should contain words having 4 letters or more.
"""

language_counter = os.getcwd()
file_path= language_counter + "/Resources/article.txt"

# Read the whole text.
def wordCloud(path):
    text_blob = open(path, encoding="utf8").read()
    text_blob = text_blob.replace("\n", " ").split(" ")
    text_blob = " ".join([word for word in text_blob if len(word)>4])
    wordcloud = WordCloud().generate(text_blob)
    plt.imshow(wordcloud)
    plt.axis("off")
    wordcloud = WordCloud(background_color="white", max_words=2000,max_font_size=40, relative_scaling=.4).generate(text_blob)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

wordCloud(file_path)


"""
4. You have been given a file ReviewID.txt.
It has 10646 records in it, each record is made up of two fields separated by a colon: like AzSn8aTOyVTUePaIQtTUYA:es .
The first field is review ID and the second field is language in which reviews has been written.
Read this file and create a bar graph showing the percentage of the reviews written in a particular language. The aim of this problem is to generate a graph using which we can do a comparative analysis of the languages used for writing reviews.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/ReviewID.txt"
file_ptr=open(dataFilePath, "r")
line_words = {}
language_counter = {}
for line in file_ptr:
    key,value = line.strip("\n").split(":")
    line_words[key]=value

count = 0
for key, value in line_words.items():
    if value in language_counter.keys():
        language_counter[value] = language_counter[value] + 1
    else:
        language_counter[value] = 1
    count += 1
percentage_value = {}
for key, value in language_counter.items():
    percentage_value[key]=round((float(value) / count) * 100, 2)

sorted_dict = sorted(percentage_value.items(), key=lambda x: x[1], reverse=True)

index = np.arange(len(sorted_dict))
barWidth = 0.35
opacity = 0.4

plt.bar(index, [value for key, value in sorted_dict], barWidth, alpha=opacity, color = 'r', align='center' )
plt.xticks(index,[key for key, value in sorted_dict])
plt.xlabel("language")
plt.ylabel("percent")
plt.show()
