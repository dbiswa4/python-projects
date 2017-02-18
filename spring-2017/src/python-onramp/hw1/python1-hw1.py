#Q1.
#Given a list like myList=[1,2,3,4]  your task is to find sum of each number with another number i.e. 1+2,1+3,1+4,2+3,2+4,3+4 .
# Write two codes, one using list comprehension and other using for loop.

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