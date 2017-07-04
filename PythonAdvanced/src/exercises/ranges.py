#Create a script that generates and prints a list of numbers from 1 to 20. Please do not create the list manually.

'''
range()  is a Python built-in function that generates a range of integers. However, range()
creates a Python range object. To get a real list object you need to use the list() function to convert the range object into a list object.
'''

my_range = range(1, 21)
print(list(my_range))


out = [x*10 for x in my_range]
print out

out = [x*10 if x%2 == 0 else x*5 for x in my_range]
print out

#Stringify
out = [str(x) for x in my_range]
print out
#Using map function
#map function iterate thru the range object and apply str to each of the elements
out = list(map(str, my_range))
print out