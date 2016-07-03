from Point import *
if __name__ == "__main__":
    print "Garbage Collection demo again..."
    
    pt1 = Point()
    pt2 = pt1
    pt3 = pt1
    print id(pt1), id(pt2), id(pt3) # prints the ids of the obejcts
    
    del pt1
    del pt2
    del pt3
    
    