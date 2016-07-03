class Networkerror(RuntimeError):
    def __init__(self, arg):
        self.args = arg

if __name__ == "__main__":
    print "User defined function...\n"
    
    try:
        raise Networkerror("Bad hostname")
    except Networkerror,e:
        print e.args
   
