if __name__ == '__main__':
    print("Beginning")

    flag = 1
    n = 1

    while flag:
        print("n : ", n)
        val1 = 3 * n * n + 5 * n + 60
        val2 = 2 * n * n * n + 2 * n + 8
        print("val1 : ", val1)
        print("val2 : ", val2)
        n += 1
        if val1 < val2:
            print("n is %d", n)
            flag = 0

        #Test breaking of loop
        #flag = 0
