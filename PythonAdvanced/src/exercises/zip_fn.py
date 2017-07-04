#Print out in each line the sum of homologous items of the two sequences.

a = [1, 2, 3]
b = (4, 5, 6)

for i, j in zip(a, b):  #zip combines two sequences
    print i + j


#print
#ab
#cd

import string
with open("letters.txt", "w") as file:
    for letter1, letter2 in zip(string.ascii_lowercase[0::2], string.ascii_lowercase[1::2]):
        file.write(letter1 + letter2 + "\n")