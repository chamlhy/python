#move function
def move(x, source, target):
	print("move No.%d from %s to %s" % (x, source, target))

def hanoi(x, source, temp, target):
	if x == 1:
		move(x, source, target)
	else:
		hanoi(x-1, source, target, temp)
		move(x, source, target)
		hanoi(x-1, source, temp, target)

if __name__ == '__main__':
	x = int(input('please input number of plate: \n'))
	source = 'A'
	temp = 'B'
	target = 'Z'
	hanoi(x, 'A', 'B', 'C')
	
