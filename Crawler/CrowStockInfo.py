import requests
import re
import bs4
from bs4 import BeautifulSoup

#获取html内容
def getHTMLText(url, code='utf-8'):
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status()
        r.encoding = code
        return r.text
    except:
        return ''

#获取股票编码列表
def getStockList(slist, html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        count = 0
        for stocks in soup.find('div', 'quotebody').find_all('ul'):
            for stock in stocks.children:
                if isinstance(stock, bs4.element.Tag):
                    if(count):
                        stype = 'sz'
                    else:
                        stype = 'sh'
                    string = stock.string
                    slist.append([string.split('(')[0], stype + string.split('(')[1].split(')')[0]])
            count += 1
    except:
        print('')

#获取一只股票的信息
def get1StockInfo(simpleInfo):
    try:
        url = 'https://gupiao.baidu.com/stock/' + simpleInfo[1] + '.html'
        html = getHTMLText(url)
        soup = BeautifulSoup(html, 'html.parser')
        oneInfo = {'股票名称' : simpleInfo[0]}
        info = soup.find('div', attrs={'class':'stock-bets'})
        keyList = info.find_all('dt')
        valueList = info.find_all('dd')
        for i in range(len(keyList)):
                key = keyList[i].text
                val = valueList[i].text
                oneInfo[key] = val
        return oneInfo
    except:
        return ''

#获取所有的股票信息并写入文件
def getAllStockInfo(slist, fpath):
    count = 0
    for simpleInfo in slist:
        try:
            oneInfo = get1StockInfo(simpleInfo)
            if oneInfo == '':
                continue
            with open(fpath, 'a', encoding='utf-8') as f:
                    f.write( str(oneInfo) + '\n' )
                    count = count + 1
                    print("\r当前进度: {:.2f}%".format(count*100/len(slist)), end="")
        except:
            count = count + 1
            print("\r当前进度: {:.2f}%".format(count*100/len(slist)),end="")
            continue 

if __name__ == '__main__':
    url = 'http://quote.eastmoney.com/stocklist.html'
    output_file = 'D:/BaiduStockInfo.txt'
    slist = []
    html = getHTMLText(url, 'GB2312')
    getStockList(slist, html)
    print('start')
    getAllStockInfo(slist, output_file)
    


