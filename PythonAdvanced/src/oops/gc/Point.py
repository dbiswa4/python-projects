class Point:
    
    #Constructor
    def __init( self, x=0, y=0):
        self.x = x
        self.y = y
    #A class can implement the special method __del__(), called a destructor, that is invoked 
    #when the instance is about to be destroyed.
    #This __del__() destructor prints the class name of an instance that is about to be destroyed
    
    def __del__(self):
        class_name = self.__class__.__name__
        print class_name, "destroyed"