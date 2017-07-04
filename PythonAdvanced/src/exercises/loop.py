import time

i = 0

while True:
    print 'Hello'
    i = i + 1
    if i > 3:
        print "End of loop"
        break
    time.sleep(1)


