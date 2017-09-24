#!/usr/bin/env python3

#Question 4

#Find Squares and add it to new list
def squareEach(nums):
    temp_list = []
    for i in nums:
        temp_list.append(i*i)
    return temp_list


#Sum list of floating point numbers
def sumList(nums):
    temp = 0.0
    for i in nums:
        temp += i
    return temp


#Coneverts to list of float numbers

def toNumbers(strList):
    for i in range(0, len(strList)):
        temp = strList[i]
        strList[i] = float(temp)


def main():
    print("The program prints the sum of squares of each line in a file.")
    fname = input("Enter the name of file with numbers: ")
    #fname = raw_input("Enter the name of file with numbers: ")
    count = 1
    f = open(fname, 'r')
    for line in iter(f):
        this_line = line.rstrip('\n')
        if len(this_line) != 0:
            strList = this_line.split()
            #print(strList)
            toNumbers(strList)
            #print(strList)
            squares = squareEach(strList)
            result = sumList(squares)
            print("Sum of squares in line", count, "is", result)
        count += 1

    f.close()


if __name__ == '__main__':
    main()