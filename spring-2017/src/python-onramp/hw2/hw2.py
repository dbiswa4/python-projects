import datetime
import math
import pandas
import sys
import random


'''
The script expects two input args.
arg1 : Fully qualified location for crimeData.csv
arg2 : Fully qualified location for timeData2.csv
'''

###Q.1
'''
Given a decimal integer number, write a function which converts this number into its binary form. Use iteration to
solve this problem. E.g. myIntNumber=5 result should be 0101.
'''

print '\nQuestion # 1'
def int_to_binary(n):
    binary_num = ""
    if n == 0:
        binary_num = "0"
    else:
        divisor, dividend, quotient, remainder = (2, n, n, 0)

        while (quotient != 0):
            remainder=dividend%divisor
            quotient = dividend/divisor
            dividend = quotient
            binary_num = str(remainder) + binary_num
    return binary_num

for n in range(10):
    print "Integer Numbe :  "+str(n) + " ## Equivalent Binary Number :  " + int_to_binary(n)


###Q.2
'''
Given an array representing binary form of a decimal integer number. Convert this into corresponding decimal number eg. [1,0,0,1] to 9.
'''
print '\nQuestion # 2'

def convert_binary_to_decimal(n):
    dec_num = 0
    for i in range(len(n)):
        dec_num += n[-i - 1] * 2 ** i
    return dec_num


print 'Decimal Number of [0, 1, 1, 1] is : ', convert_binary_to_decimal([0, 1, 1, 1])
print 'Decimal Number of [0, 0, 1, 0] is : ', convert_binary_to_decimal([0, 0, 1, 0])


###Q.3.
'''
We have a dictionary. Key of this dictionary is medicine name and values is expiry date. Write a function which takes
this dictionary as an input and print a list of all those medicine which has expired.
'''
print '\nQuestion # 3'
def get_expired_med_list(med_date):
    today = datetime.datetime.now()
    expired_meds = []
    for med in med_date.keys():
        med_manufac_date = datetime.datetime.strptime(med_date.get(med), '%b %d %Y')
        if today > med_manufac_date:
            expired_meds.append(med)
    return expired_meds


my_med_dict={
    "Abelcet":"Aug 1 2016",
    "Azithromycin":"Dec 24 2016",
    "Arava":"Jan 1 2017",
    "Arixtra":"May 31 2016",
    "Aplenzin":"Jan 3 2016",
    "Antizol":"Aug 31 2016",
    "Anadrol-50":"Nov 14 2017"
}
expired_med_list = list()
expired_med_list = get_expired_med_list(my_med_dict)
print 'Expired Medicine List : ', expired_med_list


###Q.4
'''
Write a recursive algorithm to convert a decimal integer number into its binary form e.g. myInt=9 to 1001
'''
print '\nQuestion # 4'
def convert_to_binary(num, bin):
    quotient = num
    bin1 = bin
    while quotient!=0:
        bin1 = str(quotient%2)+bin1
        quotient = int(math.floor(quotient/2))
        convert_to_binary(quotient, bin1)
    return bin1

for n in range(10):
    print 'Binary of ' + str(n) + ' is : ', convert_to_binary(n, "")


###Q.5
'''
You have been given a file "crimeData.csv", containing address and number of times a crime has happened at that address.
Your task is to write a function which takes a crime name as an input and calculates the mean of that crime i.e average
number of times that crime has happened at addresses. The function should return all the address (may be in a list)
where this crime has occurred more than the mean number of times.
'''
print '\nQuestion # 5'
def more_crime_addresses(data, crime):
    crime_mean = data[crime].mean()
    result = []
    for i in range(len(data)):
        if data[crime][i] >= crime_mean:
            result.append(data["Address"][i])
    return result

#Fully qualified path of input file expected from command line
crime_data_file = sys.argv[1]
data = pandas.read_csv(crime_data_file)
for crime in data.columns[1:]:
    result = more_crime_addresses(data, crime)
    print 'Crime  : ',crime
    print 'Result : ', result



###Q.6
'''
You have been given a file "timeData2.csv". This file contains date along with the time at which crimes has happened.
Your task is to write a function which takes a time interval and tells you which category of crime has occurred the
most in that time interval. The time interval can be taken as two separate parameter fromTime and toTime
'''
from datetime import datetime,timedelta

print '\nQuestion # 6'
def max_val_key(d):
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def get_max_crime_cat(data, fromDate, toDate):
    data.index = data.Dates
    tempData = data[(data.Dates>fromDate) & (data.Dates<toDate)]
    category_data1 = {}
    for category, category_data in tempData.groupby("Category"):
        category_data1[category] = len(category_data)
    return max_val_key(category_data1)


#Fully qualified input file expected from command line
time_data_file = sys.argv[2]
data = pandas.read_csv(time_data_file)
data.Dates = data.Dates.apply(lambda d: datetime.strptime(d, "%m/%d/%Y %H:%M"))
max_date = data.Dates.max()
min_date = data.Dates.min()
d = max_date - min_date
random_interval = random.randint(0, d.days)
from_date = min_date
to_date = min_date + timedelta(days=random_interval)
print 'Max crime category  : ', get_max_crime_cat(data, from_date, to_date)

'''
Max crime category  :  LARCENY/THEFT
'''

##################################End of Script ###################################




