def update_string():
    print "\nupdate_string()..."
    var1 = 'Hello World!'
    print "Updated String :- ", var1[:6] + 'Python'

def string_formatting():
    print "My name is %s and weight is %d kg!" % ('Captain Nemo', 65)

if __name__ == "__main__":
    
    str = 'Hello World!  '

    print str          # Prints complete string
    print str[0]       # Prints first character of the string
    print str[2:5]     # Prints characters starting from 3rd to 5th
    print str[2:]      # Prints string starting from 3rd character
    print str * 2      # Prints string two times
    print str + "TEST" # Prints concatenated string
    
    update_string()
    string_formatting()