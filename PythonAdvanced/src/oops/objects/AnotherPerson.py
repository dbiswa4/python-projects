class Person(object):

    def __init__(self, name):
        self.name = name
        ids = ()

    def assign_id(self):
        global ids
        ids = (123, 234)
        
    def print_ids(self):
        global ids
        print "ids : ", ids
        print "self.name : ", self.name
        
        