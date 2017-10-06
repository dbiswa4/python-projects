def mysteryfunction(A, right):
    if right == 1:
        print 'Value of right : 1'
        return A[1]
    else:
        val = mysteryfunction(A, right-1)
        print 'Value of val : ', val
        print 'Value of right : ', right
        if A[right] > val:
            val = A[right]
    return val

def mysteryfunction2(A, right):
    if right == 1:
        print 'right is 1'
        return A[0]
    else:
        val = mysteryfunction2(A, right-1)
        print 'Value of val : ', val
        print 'right : ', right
        if A[right - 1] > val:
            val = A[right - 1]
    return val

if __name__ == '__main__':
    print 'Begin...'
    A = [1,2,3]
    n = 3
    r = mysteryfunction2(A, n)

    print 'Final result : ', r

    A = [1, 2, 3,7,5]
    n = 5
    r = mysteryfunction2(A, n)
    print 'Final result : ', r


    A = [10, 2, 3,7,1]
    n = 5
    r = mysteryfunction2(A, n)
    print 'Final result : ', r


    A = [10, 2, 3,7,1]
    n = 4
    r = mysteryfunction(A, n)
    print 'Final result : ', r

    A = [1,2,3]
    n = 2
    r = mysteryfunction(A, n)
    print 'Final result : ', r
