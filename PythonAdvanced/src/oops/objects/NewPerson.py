class NewPerson():
    
    def __init__(self, name):
        self.name = name
        ids = ()
        dl = ()

    def assign_id(self):
        global ids
        ids = (123, 234)
        global dl
        dl = ('dl123', 'dl234')
        
    def print_ids(self):
        global ids
        print "ids : ", ids
        print "self.name : ", self.name
        
        global dl
        print "dl : ", dl
        
        