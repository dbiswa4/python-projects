def ana(s1, s2):
    for i in range(len(s1)):
        print i
        print "s1[i]  : ", s1[i]
        print "s2[-i-1] : ", s2[-i-1]
        if s1[i] != s2[-i-1]:
            return "Not Anagram"
    return "Anagram"

s1 = "dog"
s2 = "god"
print s1[1]

print ana(s1, s2)