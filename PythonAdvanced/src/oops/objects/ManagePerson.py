from Person import *
from NewPerson import *
from AnotherPerson import *
if __name__ == "__main__":
    
    p2 = NewPerson("Dipankar")
    p2.assign_id()
    p2.print_ids()
    
    #Will give error as it is expecting 2 arguments; one is 'self', which is already given and other one is 'name'
    #p1 = Person();
    
    p3 = Person("King")
    p3.assign_id()
    p3.print_ids()
    
    