def mutil(n):
	sum=n
	for i in range(1,n):
		sum *=i
	return sum

def recursion(n):
    if n==1:
        return 1
    else:
        return n*recursion(n-1)
number = int(input('请输入一个正整数：'))
result1 = mutil(number)
result2 = recursion(number)
print ("%d 的阶乘是：普通方法： %d，迭代方法 %d" % (number, result1, result2))

