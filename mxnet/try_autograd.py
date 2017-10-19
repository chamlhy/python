import mxnet.ndarray as nd
import mxnet.autograd as ag

'''-----------------------------------'''
#变量求导
x = nd.array([[1,2],[3,4]])
#开辟x的导数的内存空间
x.attach_grad()
#显式的要求MXNet记录我们需要求导的程序
with ag.record():
    y = x * 2
    z = y * x
#求导
z.backward()
#获得导数
print(x.grad)

'''----------------------------------'''
#头梯度
head_grad = nd.array([[10, 1.], [.1, .01]])
z.backward(head_grad)
print(x.grad)

'''头梯度理解
    这里的结果相当于上面的结果乘一个系数 文档原文如下：“当y是一个更大的z函数的一部分，并且我们希望求得dz/dx保存在x.grad中时，我们可以传入头梯度dz/dy的值作为backward()方法的输入参数，系统会自动应用链式法则进行计算。’
    其实意思是：这里的z和原文中的“z = y*x”不是一个概念。 或者说是u(z)，作者的意思应该是当z的函数更为复杂时（也就是u(z)），我们可以将du/dz当做参数传入z.backward。这时候x.gard即为du/dx。从某种角度上来说，即为之前的梯度乘上了系数（值为头梯度）
    '''



'''----------------------------------'''
#控制流求导
def f(a):
    b = a *2
    #b的L2范数的标量
    while nd.norm(b).asscalar() < 1000:
        b = b *2
    #b的轴上和的标量
    if nd.sum(b).asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
    
'''理解轴（axis）
    如果将三维数组的每一个二维看做一个平面（plane，X[0, :, :], X[1, :, :], X[2, :, :]），三维数组即是这些二维平面层叠（stacked）出来的结果。
    则（axis=0）表示全部平面上的对应位置，
    （axis=1），每一个平面的每一列，
    （axis=2），每一个平面的每一行。
    sum函数默认axis=()，会将数组拉成一个（1，）形状，然后求所有元素的和
    '''

#一个服从高斯分布的随机ndarray
a = nd.random_normal(shape=3)
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()
print(a.grad)
