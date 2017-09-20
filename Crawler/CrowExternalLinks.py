#此程序有以下问题：
#未进行异常处理
#未考虑请求被拒绝的情况
#未考虑进入一个没有外链也没有內链的页面的情况

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random

pages = set()
random.seed(datetime.datetime.now())

#获取页面内所有内链的列表
def getInternalLinks(bsObj, includeUrl):
    internalLinks = []
    #找出所有以‘/’开头的链接
    for link in bsObj.find_all('a', href = re.compile('^(/|.*'+includeUrl+')')):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in internalLinks:
                internalLinks.append(link.attrs['href'])
    return internalLinks

#获取页面内所有外链的列表
def getExternalLinks(bsObj, excludeUrl):
    externalLinks=[]
    # 找出所有以"http"或"www"开头且不包含当前URL的链接
    for link in bsObj.find_all('a', href = re.compile('^(http|https|www)((?!'+excludeUrl+').)*$')):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in externalLinks:
                externalLinks.append(link.attrs['href'])
    return externalLinks

def splitAddress(address):
    addressParts = re.split('https*\:\/\/', address)
    if len(addressParts) >=2:
        addressParts = addressParts[1]
    else:
        addressParts = addressParts[0]
    addressParts = addressParts.split('/')[0]
    return addressParts

#随机获取一个外链列表中的链接
def getRandomExternalLink(startingPage):
    html = urlopen(startingPage)
    bsObj = BeautifulSoup(html)
    externalLinks = getExternalLinks(bsObj, splitAddress(startingPage)[0])
    if len(externalLinks) == 0:
        internalLinks = getInternalLinks(bsObj, splitAddress(startingPage)[0])
        return getRandomExternalLink(internalLinks[random.randint(0, len(internalLinks)-1)])
    else:
        return externalLinks[random.randint(0, len(externalLinks)-1)]

def followExternalOnly(startingSite):
    externalLink = getRandomExternalLink(startingSite)
    print('随机外链是：'+externalLink)
    followExternalOnly(externalLink)

if __name__ == '__main__':
    followExternalOnly('http://oreilly.com')
            
