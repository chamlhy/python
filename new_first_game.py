import random
print("---------new game----------")
temp = input("请输入你猜测到的数据：")
num = int(temp)
answer= random.randint(1,10)
while num != answer and num!=0:
    if num>answer:
        print("猜得太大啦！！！")
    else:
        print("猜得太小啦！！！")
    temp = input("请重新输入(退出请输入0)：")
    num = int(temp)
if num == answer:
    print("恭喜猜对啦！！！")
print("游戏结束！！！")
