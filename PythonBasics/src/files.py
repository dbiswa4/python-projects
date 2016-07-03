import os

#syntax
#file object = open(file_name [, access_mode][, buffering])
def open_file():
    # Open a file
    fo = open("/Users/dbiswas76/Documents/Work/JavaRelated/experiment/Test/foo.txt", "wb")
    print "Name of the file: ", fo.name
    print "Closed or not : ", fo.closed
    print "Opening mode : ", fo.mode
    print "Softspace flag : ", fo.softspace
    # Close opend file
    fo.close()

def file_write():
    # Open a file
    fo = open("foo.txt", "wb")
    fo.write( "Python is a great language.\nYeah its great!!\n");
    fo.close()

def file_read():
    # Open a file
    fo = open("foo.txt", "r+")
    #fileObject.read([count])
    #number of bytes to be read
    str = fo.read(10);
    print "Read String is : ", str
    # Close opend file
    fo.close()
    
def file_seek():
    print "\nfile_seek() method..."
    # Open a file
    fo = open("foo.txt", "r+")
    str = fo.read(10);
    print "Read String is : ", str

    # Check current position
    position = fo.tell();
    print "Current file position : ", position

    # Reposition pointer at the beginning once again
    position = fo.seek(0, 0);
    str = fo.read(10);
    print "Again read String is : ", str
    # Close opend file
    fo.close()
    
def file_rename_delete():
    #os.rename(current_file_name, new_file_name)
    # Rename a file from test1.txt to test2.txt
    os.rename( "foo.txt", "foo2.txt" )
    
    # Delete file test2.txt
    os.remove("foo2.txt")

if __name__ == "__main__":
    print "Hello World"
    open_file()
    file_write()
    file_seek()
    