#   1. Given a decimal integer number, write a function which converts this number into its binary form. Use iteration to solve this problem. E.g. myIntNumber=5 result should be 0101.

import os

def intToBinary(x):
    binaryForm = ""
    if x==0:
        binaryForm = "0"
    else:
        divisor = 2
        dividend = x
        quotient = x
        remainder = 0
        while (quotient != 0):
            remainder=dividend%divisor
            quotient = dividend/divisor
            dividend = quotient
            binaryForm = str(remainder) + binaryForm;
    return binaryForm

for i in range(16):
    print "Binary equivalent of decimal integer number "+str(i)+ " is " +intToBinary(i)


# 2. Given an array representing binary form of a decimal integer number. Convert this into corresponding decimal number eg. [1,0,0,1] to 9.

import os

def convertBinaryToDecimal(x):
    decimalNumber = 0
    for i in range(len(x)):
        decimalNumber+=x[-i-1]*2**i
    return decimalNumber


print convertBinaryToDecimal([0,1,1,1])
print convertBinaryToDecimal([0,0,1,0])
print convertBinaryToDecimal([0,0,0,0])
print convertBinaryToDecimal([1,1,1,0])


# 3.  We have a dictionary. Key of this dictionary is medicine name and values is expiry date. Write a function which takes this dictionary as an input and print a list of all those medicine which has expired.

import datetime

def getExpiredMedList(myDict):
    todaysDate = datetime.datetime.now()
    expiredMedicineList = []
    for medicine in myDict.keys():
        medicineDate = datetime.datetime.strptime(myDict.get(medicine), '%b %d %Y')
        if todaysDate>medicineDate:
            expiredMedicineList.append(medicine)
    return expiredMedicineList


myMedDict={
    "Abelcet":"Aug 1 2016",
    "Azithromycin":"Dec 24 2016",
    "Arava":"Jan 1 2017",
    "Arixtra":"May 31 2016",
    "Aplenzin":"Jan 3 2016",
    "Antizol":"Aug 31 2016",
    "Anadrol-50":"Nov 14 2017"
}
mylist = []
mylist = getExpiredMedList(myMedDict)
print (mylist)


# 4. Write a recursive algorithm to convert a decimal integer number into its binary form e.g. myInt=9 to 1001

import math
def findBinary(number,binaryform1):
    quotient = number
    binaryform = binaryform1
    while quotient!=0:
        binaryform = str(quotient%2)+binaryform
        quotient = int(math.floor(quotient/2))
        findBinary(quotient,binaryform)
    return binaryform

print findBinary(9,"")


# 5. You have been given a file "crimeData.csv", containing address and number of times a crime has happened at that address.
# Your task is to write a function which takes a crime name as an input and calculates the mean of that crime i.e average number of times that crime has happened at addresses.
# The function should return all the address (may be in a list) where this crime has occurred more than the mean number of times.

import pandas
import os


def addressesWithMoreCrime(data,crime):
    meanOfColumn = data[crime].mean()
    result = []
    for i in range(len(data)):
        if data[crime][i] >= meanOfColumn:
            result.append(data["Address"][i])
    return result

relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/crimeData.csv"
data = pandas.read_csv(dataFilePath)
for crime in data.columns[1:]:
    result = addressesWithMoreCrime(data,crime)
    print crime
    print result



# 6. You have been given a file "timeData2.csv". This file contains date along with the time at which crimes has happened.
# Your task is to write a function which takes a time interval and tells you which category of crime has occurred the most in that time interval.
# The time interval can be taken as two separate parameter fromTime and toTime .

import os
import pandas
from datetime import datetime,timedelta
import random

def keywithmaxval(d):
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def getMaxCrimeCategory(data,fromDate,toDate):
    data.index = data.Dates
    tempData = data[(data.Dates>fromDate) & (data.Dates<toDate)]
    category_data1 = {}
    for category, category_data in tempData.groupby("Category"):
        category_data1[category] = len(category_data)
    return keywithmaxval(category_data1)



relativePath=os.getcwd()
dataFilePath=relativePath+"/Resources/timeData2.csv"
data = pandas.read_csv(dataFilePath)
data.Dates = data.Dates.apply(lambda d: datetime.strptime(d, "%m/%d/%Y %H:%M"))
# data.index = data.Dates
maxDate = data.Dates.max()
minDate = data.Dates.min()
d = maxDate-minDate
randominterval = random.randint(0,d.days)
fromDate = minDate
toDate = minDate + timedelta(days=randominterval)
print getMaxCrimeCategory(data,fromDate,toDate)




