import sys

def bottom_up(p, n):
    print('bottom_up()')

    '''
    At first position we can have three numbers 1 or 2 or 3.
First put 1 at first position and recursively call for n-1.
Then put 2 at first position and recursively call for n-2.
Then put 3 at first position and recursively call for n-3.
If n becomes 0 then we have formed a combination that compose n, so print the current combination.

    Input = 5
    First:
    1 recur(4)
        1 recur(3)
            1 recur(2)
                1 recur(1)
                    1 recur(0)  => should not be called as 0 < 1

    2nd:
    2 recur(3)
        2 recur(1)
        No more as 1 < 2



        


    '''



if __name__ == '__main__':
    print('Bottom Up - find ints')
    n = int(sys.argv[1])
    p = [1, 3, 4]
    bottom_up(p, n)
