def try_except_else():
    try:
        fh = open("testfile.txt", "w")
        fh.write("This is my test files for exception handling!!")
    except IOError:
        print "Error: can\'t find files or read data"
    else:
        print "Written content in the files successfully"
        fh.close()

def try_finally_except():
    try:
        fh = open("testfile", "w")
        try:
            fh.write("This is my test files for exception handling!!")
        finally:
            print "Going to close the files"
            fh.close()
    except IOError:
        print "Error: can\'t find files or read data"

def argument_of_exception_temp_convert(var):
    print "\nargument_of_exception_temp_convert(var)..."
    try:
        return int(var)
    except ValueError, Argument:
        print "The argument does not contain numbers\n", Argument

def raise_exception(level):
    if level < 1:
        raise "Invalid level!", level
        # The code below to this would not be executed
        # if we raise the exception    

if __name__ == "__main__":
    print "Exception handling"
    try_except_else()
    argument_of_exception_temp_convert("abc")
    raise_exception(0)
    
    