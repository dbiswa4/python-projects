def firstTenReverse1(nums):
    for i in range(len(nums) - 1, -1, -1):
        print(nums[i])

def firstTenReverse2(nums, count=10):
    for i in range(count - 1, -1, -1):
        print(nums[i])

def firstTenReverse3(nums):
    for i in range(len(nums) - 1, -1):
        print(nums[i])

def firstTenReverse4():
    print("4th")
    for i in range(20,1,-2):
        print(i)

def firstTenReverse99(nums):
    print("99th")
    while (len(nums) > 0):
        print(nums.pop())


if __name__ == '__main__':
    print 'Print in reverse oreder'
    nums = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    firstTenReverse2(nums)
    firstTenReverse1(nums)
    firstTenReverse3(nums)
    firstTenReverse4()

    #Be careful
    #Using pop() in the method. It will make the list empty.
    #This is not a good fearture in Python where variable namespace conflicts
    #Ideally the method should have made a copy of the do the operation. Looks like Python works
    #as per pass-by-reference
    firstTenReverse99(nums)
