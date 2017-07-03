def read_csv_file(temp_file_path):
    with open(temp_file_path, 'r') as f:
        #this_line = f.read().rstrip('\n')
        this_line = f.read()
        print "this_line: " + str(this_line)
        print 'Finished Printing'


def read_file_lines(temp_file_path):
    print '\n...In read_file_lines() method...\n'
    f = open(temp_file_path, 'r')
    for line in iter(f):
        print line.rstrip('\n')
        this_line = line.rstrip('\n')
        print 'len : ', len(this_line)
        if len(this_line) != 0:
            print 'Line Printed'

    f.close()

def read_file_lines_2(temp_file_path):
    print '\n...In read_file_lines_2() method...\n'
    with open(temp_file_path, 'r') as f:
        #this_line = f.read().rstrip('\n')
        #this_line = f.read().splitlines()
        this_line = f.readline()
        print "this_line: " + str(this_line)
        print 'Finished Printing'

def read_file_lines_3(temp_file_path):
    print '\n...In read_file_lines_3() method...\n'
    with open(temp_file_path, 'r') as f:
        for line in f:
            print line
            print line.rstrip('\n')
            print line

if __name__ == '__main__':
    print 'Test File Read'
    read_csv_file('./file1.csv')

    read_file_lines('./file2.csv')

    read_file_lines_2('./file3.csv')

    read_file_lines_3('./file4.csv')
