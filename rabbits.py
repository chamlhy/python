#iterate
def feb(n):
    n1 = 1
    n2 = 1
    n3 = 1
    if n < 1:
        print('input error!')
        return -1
    while (n-2) > 0:
        n3 = n2 + n1
        n1 = n2
        n2 = n3
        n -= 1
    return n3

# recurrence   
def feb1(n):
    if n < 1:
        print('input error!')
        return -1
    if n == 1 or n == 2:
        return 1
    if n > 2:
        return feb1(n-1)+feb1(n-2)

if __name__ == '__main__':
    result = feb(20)
    result1 = feb1(20)
    if result == result1:
        print("After 20 months, rabbit's number will be %d. " % result)
    else:
        print("computation error!")
        
