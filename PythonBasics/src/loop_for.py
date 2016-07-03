def for_loop():
    print "\nfor_loop()..."
    for letter in 'Python':     # First Example
        print 'Current Letter :', letter

    fruits = ['banana', 'apple',  'mango']
    for fruit in fruits:        # Second Example
        print 'Current fruit :', fruit

    print "Good bye!"

def for_index():
    print "\nfor_index()..."
    fruits = ['banana', 'apple',  'mango']
    print "How many elements in fruits list : ", len(fruits)
    for index in range(len(fruits)):
        print 'Current fruit :', fruits[index]

    print "Good bye!"
    
def for_range():
    print "\nfor_range()..."
    for num in range(10,20):  #to iterate between 10 to 20
        for i in range(2,num): #to iterate on the factors of the number
            if num%i == 0:      #to determine the first factor
                j=num/i          #to calculate the second factor
                print '%d equals %d * %d' % (num,i,j)
                break #to move to the next number, the #first FOR
            else:                  # else part of the loop
                print num, 'is a prime number'

def for_nested():
    print "\nfor_nested()..."
    i = 2
    while(i < 100):
        j = 2
        while(j <= (i/j)):
            if not(i%j): break
            j = j + 1
        if (j > i/j) : print i, " is prime"
        i = i + 1

    print "Good bye!"

if __name__ == "__main__":
    print "Hello World"
    for_loop()
    for_index()
    for_range()
    for_nested()