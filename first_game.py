print("------------A new game-------------")
temp = input("不妨猜一下我想的是哪个数字：")
guess = int(temp)
if guess == 4:
    print("你是我肚里的蛔虫吧！！！")
    print("猜中也没有奖励！")
else:
    print("猜错啦，我想的是4！！")
print("Game Over")
