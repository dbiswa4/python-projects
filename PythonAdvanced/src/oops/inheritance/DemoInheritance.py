from Child import *
from Parent import *
if __name__ == "__main__":
    print "Demo Inheritance..."
    
    c = Child()          # instance of child
    c.childMethod()      # child calls its method
    c.parentMethod()     # calls parent's method
    c.setAttr(200)       # again call parent's method
    c.getAttr()          # again call parent's method

    #Create another child object    
    c1 = Child()
 
    #Note:
    #Why it does not display the method names in below case? But it does for above case.
    
    #c1. 
    
    #Create Parent object
    p = Parent()
    p.parentMethod()
    #AttributeError: Parent instance has no attribute 'childMethod'
    #p.childMethod()
    
    #Run time polymorphism - not Python cup of tea, it seems 
    #Parent p2 = Child()
    #p2.childMethod()
    