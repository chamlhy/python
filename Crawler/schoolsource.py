import requests
from bs4 import BeautifulSoup
import bs4

def getHTMLText(url):
    try:
        r = requests.get(url, timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text  
    except:
        print("something error!")
        return " "

def fillUnivList(alllist, html):
    soup = BeautifulSoup(html, 'html.parser')
    for school in soup.find('tbody').children:
        if isinstance(school, bs4.element.Tag):
            nodes = school('td')
            alllist.append([nodes[0].string, nodes[1].string, nodes[3].string])

def printUnivList(alllist, num):
    tplt = '{0:<10}\t{1:{  3}<6}\t{2:<10}'
    print(tplt.format('排名', '学校名称', '总分', chr(12288)))
    for i in range(num):
        u = alllist[i]
        print(tplt.format(u[0], u[1], u[2], chr(12288)))

if __name__ == '__main__':
    alllist = []
    url = 'http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html'
    html = getHTMLText(url)
    fillUnivList(alllist, html)
    printUnivList(alllist, 10)
    
