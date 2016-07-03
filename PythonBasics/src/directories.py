import os

def make_dir():
    os.mkdir("newdir")
    
def change_dir():
    # Changing a directory to "/home/newdir"
    os.chdir("/home/newdir")

def current_dir():
    # This would give location of the current directory
    os.getcwd()

def remove_dir():
    # This would  remove "/tmp/test"  directory.
    os.rmdir( "/tmp/test"  )
        
if __name__ == "__main__":
    print "Hello World"
