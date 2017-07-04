letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
print(letters[-2])
print(letters[-2:])

'''
The complete syntax of list slicing is [start:end:step] . When you don't pass a step, Python assumes the step is 1.
[:]  itself means get everything from start to end. So, [::2]  means get everything from start to end at a step of two.
'''
print(letters[::2])

#output
#['a', 'c', 'e', 'g', 'i']
