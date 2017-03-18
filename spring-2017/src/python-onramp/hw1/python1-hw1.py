#!/usr/bin/env python

import re

#######################################################################################################################
#Q1.
#Given a list like myList=[1,2,3,4]  your task is to find sum of each number with another number i.e. 1+2,1+3,1+4,2+3,2+4,3+4 .
# Write two codes, one using list comprehension and other using for loop.

print '\n******Q1....'

myList=[1,2,3,4]
print 'myList : ', myList

outList=[]

for i in range(0,myList.__len__() - 1):
    #print '\nmyList[i] : ', myList[i]
    for x in range(i+1,myList.__len__()):
        #print 'myList[x] : ', myList[x]
        outList.append(int(myList[i]) + int(myList[x]))

print 'outList : using Loop\n', outList

##Very important
##First for loop indicates to outer loop, the next one is inner loop
##Little contradictory to common sense
print 'outList : List Comphrehension'
#print [int(myList[i]) + int(myList[x]) for x in range(i+1, myList.__len__()) for i in range(0,myList.__len__() - 1) ]
print [int(myList[i]) + int(myList[x]) for i in range(0,myList.__len__() - 1) for x in range(i+1, myList.__len__()) ]



#######################################################################################################################
#Q2.
#2.Given a list write a code to detect if there is a duplicate element present in the list or not. Print yes or no.
# Write two codes, one using '==' operator and other using exclusive or operator '^'.

print '\n\n******Q2....'
print '\nFind duplicate in list with =='
myList=[1,2,3,4,4]
dup = False
i = 0
while i < len(myList) - 1 and not dup:
    j = i + 1
    #print 'myList[i] : ', myList[i]
    while j < len(myList) and not dup:
        #print 'myList[j] : ', myList[j]
        if myList[i] == myList[j]:
            print 'yes'
            dup = True
        j += 1
    i += 1

if not dup:
    print 'No'

#With X-or (^)
print '\nFind duplicate in list with X-or'
myList=[1,2,3,4,4]
dup = False
i = 0
while i < len(myList) - 1 and not dup:
    j = i + 1
    #print 'myList[i] : ', myList[i]
    while j < len(myList) and not dup:
        #print 'myList[j] : ', myList[j]
        if not myList[i]^myList[j]:
            print 'yes'
            dup = True
        j += 1
    i += 1

if not dup:
    print 'no'


#######################################################################################################################
#Q3.
#Given below is a 2D matrix, create a list of all the odd numbers present in the matrix. Also sort the list in descending order.
print '\n\n******Q3....'
myMatrix = [[1, 2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]]

print 'myMatrix : ', myMatrix

odds = []
for row in myMatrix:
    for col in row:
        if col % 2 != 0:
            odds.append(col)

odds_sorted = sorted(odds, reverse=True)
print 'Sorted odds : \n', odds_sorted


#Using List Comprehension
odds = [x for myMatrix in myMatrix for x in myMatrix if x%2 != 0]
odds_sorted = sorted(odds, reverse=True)
print '\nSorted odds with list comphrehension : \n', odds_sorted

#######################################################################################################################
#4
#Given below is a 2D matrix, create a list of squares of all the even numbers present in the matrix.
print '\n\n******Q4....'
myMatrix = [[1, 2, 'aa',3, 4],
            ['dd',5, 6, 7],
            [8, 9, 10,'cc']]
sq = []
for row in myMatrix:
    for col in row:
        if isinstance(col, int):
            if col % 2 == 0:
                sq.append(col**2)

print 'Square of all Even number : \n', sq

#Using List Comprehension
sq = [x*x for myMatrix in myMatrix for x in myMatrix if isinstance(x, int) and x%2 == 0]
print '\nSquare of all Even number with List Comphrehension: \n', sq

#######################################################################################################################
#5. Given below is a 2D matrix, create a list of squares of all the prime numbers present in the matrix.
# (Hint: use 6k+1 or 6k-1 formula)
print '\n\n******Q5....'
myMatrix = [[21, 22, 23, 4, 16, 17, 18, 19],
            [5, 6, 7, 14, 15, 20, 1, 2, 3],
            [8, 9, 10, 11, 12, 13]]

primes = []
prime_sqaure = []
for row in myMatrix:
    for col in row:
        col *= 1.0
        if col % 2 == 0 and col != 2 or col % 3 == 0 and col != 3:
            continue
        for b in range(1, int((col ** 0.5 + 1) / 6.0 + 1)):
            if col % (6 * b - 1) == 0:
                continue
            if col % (6 * b + 1) == 0:
                continue
        primes += [int(col)]
        prime_sqaure += [int(col**2)]


print 'primes : ', primes
print 'Square of prime : ', prime_sqaure


#######################################################################################################################
#6. Make a dictionary of all those words, from the given paragraph, which are having 4 or more letters in it .
#Key of the dictionary should be word and value should be the number of times that word has appeared in the paragraph.
# eg. {"feminist":3,"part":2,"campaign":1}

print '\n\n******Q6....'
mySentence="It's the Spice Girls but not as you know them. Twenty years after it was first released, this famous girl " \
           "power anthem has been given a 21st century feminist makeover. The new video is part of Project Everyone's " \
           "campaign to improve the lives of women and girls everywhere, calling for an end to violence against girls, " \
           "quality education for all and equal pay for equal work."

words_count = {}
#Clean up the words
words = re.findall(r'[^\s!,.?":;0-9]+', mySentence)

for word in words:
    if len(word) >= 4 :
        if words_count.has_key(word):
            words_count[word] += 1
        else:
            words_count[word] = 1

print 'Words count : \n', words_count


#######################################################################################################################
#7. Given a list, multiply all the elements of the list by 2 without using any arithmetic operator. Hint: use bitwise operator

print '\n\n******Q7....'
myList = [1,2,3,6,8]

print 'My List : \n', myList

newList = [x << 1 for x in myList]

print 'Multiplied List : \n', newList


#######################################################################################################################
#8.Given below are two 2D matrix, add them element wise to form a third 2D matrix and print the resultant matrix.

print '\n\n******Q8....'
myMatrix1 = [[1, 2, 3, 4],
            [5, 6, 7,6],
            [8, 9, 10,4]]

myMatrix2 = [[3, 1, 1, 4],
            [7, 7, 7,7],
            [8, 9, 10,11]]


result = [[myMatrix1[i][j] + myMatrix2[i][j]  for j in range(len(myMatrix1[0]))] for i in range(len(myMatrix1))]

print 'Resulted Matrix : \n', result


print '*****End of Script*****'

######End of Script#######