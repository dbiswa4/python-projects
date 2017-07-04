# 1. You have been given a barGraph.csv file. Using the data of this file you have to draw a bar graph showing all 8 emotions corresponding to each business.

import pandas
import os
import matplotlib.pyplot as plt
import numpy as np

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/barGraph.csv"
data = pandas.read_csv(dataFilePath)

index = np.arange(len(data.enjoyment))
barWidth = 0.1
opacity = 0.4

a = data.columns

count = 0
for column in a.tolist()[1:]:
    plt.bar(index+count*barWidth,data[column].tolist(),barWidth,label=column)
    count+=1
    if count==int(len(a.tolist())/2):
        plt.xticks(index+count*barWidth,data['Business'].tolist())

plt.legend()
loc, labels = plt.xticks()
plt.setp(labels,rotation=-8)
plt.show()

# 2. Using the data present in barGraph.csv file generate pie-chart showing percentage of emotions for each business.

import pandas
import os
import matplotlib.pyplot as plt
import numpy as np

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/barGraph.csv"
data = pandas.read_csv(dataFilePath)

labels = data.columns[1:].tolist()
count = 1
a, b = plt.subplots(3,2)
for i in range(3):
    for j in range(2):
        # count+=1
        b[i,j].pie(data.loc[i+j][1:].tolist(),autopct='%1.1f%%', radius=0.8)
        b[i,j].set_title(data.loc[i+j][0])
        b[i,j].legend(labels,loc="upper left", fontsize=6)
plt.show()


# 3. Generate a word cloud of your favorite news article or story or anything. This word cloud should contain words having 4 letters or more.

import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

d = os.getcwd()
filepath=d+"/Resources/article.txt"

# Read the whole text.
def wordCloud(path):
    text = open(path, encoding="utf8").read() #read the entire file in one go
    text = text.replace("\n", " ").split(" ")
    text = " ".join([word for word in text if len(word)>4])
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud)
    plt.axis("off")
    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(background_color="white", max_words=2000,max_font_size=40, relative_scaling=.4).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

wordCloud(filepath)


# 4. You have been given a file ReviewID.txt. It has 10646 records in it, each record is made up of two fields separated by a colon: like AzSn8aTOyVTUePaIQtTUYA:es . The first field is review ID and the second field is language in which reviews has been written. Read this file and create a bar graph showing the percentage of the reviews written in a particular language. The aim of this problem is to generate a graph using which we can do a comparative analysis of the languages used for writing reviews.
import os
import matplotlib.pyplot as plt
import numpy as np

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/ReviewID.txt"
filePointr=open(dataFilePath,"r")
v = {}
d = {}
f = {}
for line in filePointr:
    key,value = line.strip("\n").split(":")
    v[key]=value

count = 0
for key, value in v.items():
    if value in d.keys():
        d[value] = d[value] + 1
    else:
        d[value] = 1
    count += 1

for key, value in d.items():
    f[key]=round((float(value)/count)*100, 2)

sorted_dict = sorted(f.items(),key=lambda x: x[1],reverse=True)

index = np.arange(len(sorted_dict))
barWidth = 0.35
opacity = 0.4

plt.bar(index, [value for key, value in sorted_dict], barWidth, alpha=opacity, color = 'r', align='center' )
plt.xticks(index,[key for key, value in sorted_dict])
plt.xlabel("Review Language")
plt.ylabel("Percentage")
plt.show()
