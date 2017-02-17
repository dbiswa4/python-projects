
myList = [1,'a',0,"hi",8,3,'o']

squaredList = [ e**2 for e in myList if isinstance(e,int)  ] #list comprehension

print squaredList

M=[x for x in myList if isinstance(x,int) and x % 2 == 0]

print M


myMatrix=[[1,2,3,4],
[5,6,7],
[8,9,10]]

print myMatrix

#convert this 2D matrix into a list

list=[i for row in myMatrix for i in row ]

print list