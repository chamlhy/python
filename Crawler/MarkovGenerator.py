from urllib.request import urlopen
from random import randint

def wordListSum(wordList):
    sum = 0
    for word, value in wordList.items():
        sum += value
    return sum

def retrieveRandomWord(wordList):
    randIndex = randint(1, wordListSum(wordList))
    for word, value in wordList.items():
        randIndex -= value
        if randIndex <= 0:
            return word

def buildWordDict(text):
    #剔除换行符和引号
    text = text.replace('\n', ' ')
    text = text.replace('\"', '')

    #保证每个标点符号都被视作一个单词
    #这样不会被剔除
    punctuation = [',', '.', ':', ';']
    for symbol in punctuation:
        text = text.replace(symbol, ' ' + symbol + ' ')

    words = text.split(' ')
    #过滤空词
    words = [word for word in words if word != '']

    #形成字典
    wordDict = {}
    for i in range(1, len(words)):
        if words[i-1] not in wordDict:
            wordDict[words[i-1]] = {}
        if words[i] not in wordDict[words[i-1]]:
            wordDict[words[i-1]][words[i]] = 0
        wordDict[words[i-1]][words[i]] = wordDict[words[i-1]][words[i]] + 1
    return wordDict

if __name__ == '__main__':
    text = str(urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt')
               .read(), 'utf-8')
    wordDict = buildWordDict(text)

    #生成长度为50的马尔可夫链
    length = 50
    chain = ''
    currentWord = 'The'
    for i in range(0, length):
        chain += currentWord+' '
        currentWord = retrieveRandomWord(wordDict[currentWord])
    print(chain)
