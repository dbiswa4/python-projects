#!/usr/bin/env python

def countDigits(line):
    cnt = 0
    for letter in line:
        if letter.isdigit():
            cnt += 1
    return cnt


if __name__ == '__main__':
    print("The program prints the number of digits in each line of a file.")

    input_file_name = raw_input("Enter the name of input text file: ")
    print input_file_name

    line_count = 1
    f = open(input_file_name, 'r')
    for line in iter(f):
        this_line = line.rstrip('\n')
        if len(this_line) != 0:
            digitCount = countDigits(this_line)
            print("There are " + str(digitCount) + " digits in line " + str(line_count))
        line_count += 1
    f.close()


