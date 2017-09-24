#!/usr/bin/env python3

#Question 4
def squareEach(nums):
    temp_list = []
    for i in nums:
        temp_list.append(float(i)*float(i))


def sumList(nums):
    temp = 0.0
    for i in nums:
        temp += float(i)
    return temp



def toNumbers(strList):
    temp_list=[]
    for i in strList:
        temp_list.append(float(i))



def main():
    print("The program prints the sum of squares of each line in a file.")
    fname = input("Enter the name of file with numbers: ")
    #fname = raw_input("Enter the name of file with numbers: ")
    count = 1
    f = open(fname, 'r')
    for line in iter(f):
        this_line = line.rstrip('\n')
        if len(this_line) != 0:
            temp = this_line.split()
            print(temp)
            toNumbers(temp)
            squareEach(temp)
            result = sumList(temp)
            print("Sum of squares in line", count, "is", result)
        count += 1

    f.close()



if __name__ == '__main__':
    main()