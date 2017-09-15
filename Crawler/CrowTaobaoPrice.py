import requests
import re

#获取html内容
def getHTMLText(url):
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ''

#解析html内容
def parsePage(ilt, html):
    try:
        plt = re.findall(r'\"view_price\"\:\"[\d\.]*\"', html)
        tlt = re.findall(r'\"raw_title\"\:\".*?\"', html)
        for i in range(len(plt)):
            price = eval(plt[i].split(':')[1]) #eval去除字符串外层的单引号或者双引号
            title = eval(tlt[i].split(':')[1])
            ilt.append([price, title])
    except:
        print('')

def printGoodsList(ilt):
    tplt = "{:4}\t{:8}\t{:20}"
    print(tplt.format('序号', '价格', '商品名称'))
    count = 0
    for good in ilt:
        count += 1
        print(tplt.format(count, good[0], good[1]))

def main():
    goods = input('请输入要搜索的商品名称：')
    depth = 2
    start_url = 'https://s.taobao.com/search?q=' + goods
    infoList = []
    for i in range(depth):
        try:
            url = start_url + '&s=' + str(44 * i)
            html = getHTMLText(url)
            parsePage(infoList, html)
        except:
            continue
    printGoodsList(infoList)

if __name__ == '__main__':
    main()
        
