# -*- coding: utf-8 -*-
import requests
import time

#通用代码框架
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        print("something error!")
        return "产生异常"

if __name__ == "__main__":
    url = "http://www.baidu.com"
    #print(getHTMLText(url))
    time_start = time.time()
    for i in range(100):
        getHTMLText(url)
    time_end = time.time()
    print("总计用时：", time_end - time_start)
