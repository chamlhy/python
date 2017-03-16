Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 25 2016, 22:01:18) [MSC v.1900 32 bit (Intel)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> def add(num1,num2):
	return (num1+num2)

>>> add(3,5)
8
>>> add.__doc__
>>> def try(n1):
	
SyntaxError: invalid syntax
>>> def trytry(n1):
	'wendang'
	#zhushi
	return (n1)

>>> trytry('ajgf')
'ajgf'
>>> trytry.__doc__
'wendang'
>>> help(trytry)
Help on function trytry in module __main__:

trytry(n1)
    wendang

>>> print.__doc__
"print(value, ..., sep=' ', end='\\n', file=sys.stdout, flush=False)\n\nPrints the values to a stream, or to sys.stdout by default.\nOptional keyword arguments:\nfile:  a file-like object (stream); defaults to the current sys.stdout.\nsep:   string inserted between values, default a space.\nend:   string appended after the last value, default a newline.\nflush: whether to forcibly flush the stream."
>>> help(print)
Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
    
    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.
    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.

>>> def test(*params):
	print('长度：', len(params))
	print ('first:',params[0])

	
>>> test(1,2,3,4,5)
长度： 5
first: 1
>>> def test1(name, *params):
	print(name)
	print('长度：', len(params))
	print ('first:',params[0])

	
>>> test1(wo, 1, 2, 3, 4)
Traceback (most recent call last):
  File "<pyshell#24>", line 1, in <module>
    test1(wo, 1, 2, 3, 4)
NameError: name 'wo' is not defined
>>> test1('wo', 1, 2, 3, 4)
wo
长度： 4
first: 1
>>> def test1(*params, name):
	print(name)
	print('长度：', len(params))
	print ('first:',params[0])

	
>>> test1(1, 2, 3, 4, name='wo')
wo
长度： 4
first: 1
>>> def test2(*params, name):
	print(name)
	print('长度：', len(params))
	print ('first:',params[0])

	
>>> test2(1, 2, 3, 4, name='wo')
wo
长度： 4
first: 1
>>> test2(1, 2, 3, 4, 'wo')
Traceback (most recent call last):
  File "<pyshell#32>", line 1, in <module>
    test2(1, 2, 3, 4, 'wo')
TypeError: test2() missing 1 required keyword-only argument: 'name'
>>> def test2(*params, name):
	print(name)
	print('长度：', len(params), end=' ')
	print ('first:',params[0])

	
>>> test2(1, 2, 3, 4, name='wo')
wo
长度： 4 first: 1
>>> 
