from Employee import *
if __name__ == "__main__":
    print "Hello..."
    "This would create first object of Employee class"
    emp1 = Employee("Zara", 2000)
    "This would create second object of Employee class"
    emp2 = Employee("Manni", 5000)
    
    emp1.displayEmployee()
    emp2.displayEmployee()
    print "Total Employee %d" % Employee.empCount
    
    #Note:
    #You can add, remove, or modify attributes of classes and objects at any time 
    #Very different from Java.
    #Why do we need this feature? If we add attributes to an object from here and there, will it not be a problem to 
    #manage it?
    
    emp1.age = 7  # Add an 'age' attribute.
    emp1.age = 8  # Modify 'age' attribute.
    #del emp1.age  # Delete 'age' attribute.

    print "hasattr(emp1, 'age') : ", hasattr(emp1, 'age')    # Returns true if 'age' attribute exists
    getattr(emp1, 'age')    # Returns value of 'age' attribute
    setattr(emp1, 'age', 8) # Set attribute 'age' at 8
    delattr(emp1, 'age')    # Delete attribute 'age'
    
    
    #Built-In Class Attributes
    
    print "Employee.__doc__:", Employee.__doc__     #Class documentation string or none, if undefined. 
    print "Employee.__name__:", Employee.__name__   #Class name   
    print "Employee.__module__:", Employee.__module__ #Module name in which the class is defined. This attribute is "__main__" in interactive mode. 
    print "Employee.__bases__:", Employee.__bases__     #A possibly empty tuple containing the base classes, in the order of their occurrence in the base class list.
    print "Employee.__dict__:", Employee.__dict__       #Dictionary containing the class's namespace.









