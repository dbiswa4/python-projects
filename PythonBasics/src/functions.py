def changeme( mylist ):
    #"This changes a passed list into this function"
    '''All parameters (arguments) in the Python language are passed by reference. It means if you 
    change what a parameter refers to within a function, the change also reflects back in the calling function.'''
    mylist.append([1,2,3,4]);
    print "Values inside the function: ", mylist
    return #A return statement with no arguments is the same as return None.

def iterate_list(mylist):
    for num in mylist:
        print "num : ", num
        
def printinfo(name, age):
    #Keyword arguments
    #Order of arguments does not matter
    print "\nprintinfo(name, age)..."
    "This prints a passed info into this function"
    print "Name: ", name
    print "Age ", age
    return;

def printinfo_def_arg( name, age = 35 ):
    #Default arguments
    "This prints a passed info into this function"
    print "Name: ", name
    print "Age ", age
    return
def printinfo_var_arg( arg1, *vartuple ):
    "This prints a variable passed arguments"
    print "Output is: "
    print arg1
    for var in vartuple:
        print var
    return
 
    
if __name__ == "__main__":
    print "Main method...\n"
    mylist = [10,20,30]
    changeme( mylist )
    print "Values outside the function: ", mylist
    print "mylist[0] : ", mylist[0]
    print "mylist[3][0] : ", mylist[3][0]
    iterate_list(mylist)
    
    # Now you can call printinfo function
    printinfo( age=50, name="miki" )
    
    # Now you can call printinfo function
    printinfo_def_arg( age=50, name="miki" )
    printinfo_def_arg( name="miki" )
    
    printinfo_var_arg("Mike", "Mouse", 4)