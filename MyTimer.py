import time as t

class MyTimer:
    #初始化计时器状态
    def __init__(self):
        self.statu = 0
        self.unit = ['年','月','天','小时','分钟','秒']

    #定义魔法方法
    def __str__(self):
        if self.statu == 2:
            return self.prompt
        elif self.statu == 1:
            return '正在计时！'
        else:
            return '计时还未开始！'

    __repr__ = __str__

    def __add__(self, other):
        sum = []
        sumout = '总共运行了'
        for i in range(6):
            sum.append(self.lasted[i] + other.lasted[i])
            if sum[i]:
                sumout += str(sum[i]) + self.unit[i]
        return sumout

    #开始计时
    def start(self):
        if self.statu == 0 or self.statu == 2:
            self.statu = 1
            self.begin = t.localtime()
            print('计时开始！')
        else:
            print('计时已经开始！')

    #停止计时
    def stop(self):
        if self.statu == 1:
            self.statu = 2
            self.end = t.localtime()
            print('计时结束！')
            self._calc()
        elif self.statu == 0:
            print('计时尚未开始！')
        else:
            print('计时已经结束！')

    #内部方法，计算运行时间
    def _calc(self):
        self.lasted = []
        self.prompt = "总共运行了"
        for i in range(6):
            self.lasted.append(self.end[i] - self.begin[i])
            if self.lasted[i]:
                self.prompt += str(self.lasted[i]) + self.unit[i]
        print(self.prompt)
        
        

        
        
