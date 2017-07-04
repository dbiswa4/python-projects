c = 1

def foo():
    c = 2
    return c

c = 3
print foo() #fn will return the value of local variable c which is 2


def foo():
    global x    #make x a global variable
    x = 1
    return x
foo()
print(x)