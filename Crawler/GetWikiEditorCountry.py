from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import re
import json

random.seed(datetime.datetime.now())

def getLinks(articleUrl):
    try:
        html = urlopen('http://en.wikipedia.org'+articleUrl)
    except:
        return []
    bsObj = BeautifulSoup(html)
    return bsObj.find('div', {'id':'bodyContent'}).find_all('a',
                                                            href=re.compile('^(/wiki/)((?!:).)*$'))

def getHistoryIPs(pageUrl):
    pageUrl = pageUrl.replace('/wiki/', '')
    historyUrl = 'https://en.wikipedia.org/w/index.php?title=' + pageUrl + '&action=history'
    print('history url is: ' + historyUrl)
    try:
        html = urlopen(historyUrl)
    except:
        return set()
    bsObj = BeautifulSoup(html)
    ipAddresses = bsObj.find_all('a', {'class':'mw-anonuserlink'})
    addressList = set()
    for ipAddress in ipAddresses:
        addressList.add(ipAddress.get_text())
    return addressList

def getCountry(ipAddress):
    try:
        response = urlopen('http://freegeoip.net/json/' + ipAddress).read().decode('utf-8')
    except:
        return ''
    responseJson = json.loads(response)
    return responseJson.get('country_code')
    

if __name__ == '__main__':
    links = getLinks('/wiki/Python_(programming_language)')

    while(len(links) > 0):
        for link in links:
            print('-----------------------------')
            historyIPs = getHistoryIPs(link.attrs['href'])
            for historyIP in historyIPs:
                print(historyIP)
                country = getCountry(historyIP)
                if country != '':
                    print(historyIP + ' is from ' + country)
