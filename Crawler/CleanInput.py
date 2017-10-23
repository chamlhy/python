from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import OrderedDict
import re
import string

def cleanInput(input):
    input = re.sub('\n+', ' ', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('\[[\d]*\]', ' ', input)
    input = bytes(input, 'UTF-8')
    input = input.decode('ascii', 'ignore')
    cleanInput = []
    input = input.split(' ')
    for item in input:
        item = item.strip(string.punctuation)
        if len(item) > 1 or (item.lower() == 'a' or item.lower() == 'i'):
            cleanInput.append(item)
    return cleanInput

def ngrams(input, n):
    input = cleanInput(input)
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

if __name__ == '__main__':
    html = urlopen('https://en.wikipedia.org/wiki/Python_(programming_language)')
    bsObj = BeautifulSoup(html, 'html.parser')
    content = bsObj.find('div', {'id':'mw-content-text'}).get_text()
    ngrams = ngrams(content, 2)
    print('2-grams count is : ' + str(len(ngrams)))
    print(ngrams[:20])
    #这里的ngrams应该是记录了频率的字典类型
    #ngrams = OrderedDict(sorted(ngrams.items(), key = lambda t: t[1], reverse = True))
    #print('ordered count is :' + str(len(ngrams)))
    #print(ngrams[:20])
        
    
