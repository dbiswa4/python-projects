class Vector:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return 'Vector (%d, %d)' % (self.a, self.b)
   
   
    #You could, however, define the __add__ method in your class to perform vector 
    #addition and then the plus operator would behave as per expectation
    def __add__(self,other):
        return Vector(self.a + other.a, self.b + other.b)