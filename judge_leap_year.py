print("---------------闰年判断程序----------------")
temp = input("请输入一个年份(退出请输入0)：")
year = 1994
if temp.isdigit() == False:
    temp = input("请重新输入符合规则的数字：")
year = int(temp)
while year != 0:
    if (year%4 == 0 and year%100 != 0) or (year%400 == 0):
        print("这是一个闰年")
    else:
        print("这不是一个闰年")
    temp = input("是否继续输入，是则输入年份，否则输入0:")
    if temp.isdigit() == False:
        temp = input("请重新输入符合规则的数字：")
    year = int(temp)
print("程序结束")
