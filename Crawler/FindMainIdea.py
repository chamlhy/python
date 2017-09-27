from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import OrderedDict
import re
import string
#import operator

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
    output = {}
    for i in range(len(input)-n+1):
        if isCommon(input[i:i+n]):
            pass
        else:
            ngramTemp = ' '.join(input[i:i+n])
            if ngramTemp not in output:
                output[ngramTemp] = 0
            output[ngramTemp] += 1
    return output

def isCommon(ngram):
    commonWords = ["the", "The", "be", "and", "of", "a", "in", "to", "have", "it",
                   "i", "that", "for", "you", "he", "with", "on", "do", "say", "this",
                   "they", "is", "an", "at", "but","we", "his", "from", "that", "not",
                   "by", "she", "or", "as", "what", "go", "their","can", "who", "get",
                   "if", "would", "her", "all", "my", "make", "about", "know", "will",
                   "as", "up", "one", "time", "has", "been", "there", "year", "so",
                   "think", "when", "which", "them", "some", "me", "people", "take",
                   "out", "into", "just", "see", "him", "your", "come", "could", "now",
                   "than", "like", "other", "how", "then", "its", "our", "two", "more",
                   "these", "want", "way", "look", "first", "also", "new", "because",
                   "day", "more", "use", "no", "man", "find", "here", "thing", "give",
                   "many", "well", "It", "I"]
    for word in ngram:
        if word in commonWords:
            return True
    return False

def getFirstSentence(ngram, content):
    sentences = content.split('.')
    for sentence in sentences:
        if ngram in sentence:
            return sentence
    return ''

if __name__ == '__main__':
    content = str(
        urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt').read(),
        'utf-8')
    ngrams = ngrams(content, 2)
    sortedNGrams = sorted(ngrams.items(), key = lambda t : t[1], reverse = True)
    print('ordered count is :' + str(len(sortedNGrams)))
    print(sortedNGrams[:20])
    for word in sortedNGrams:
        if word[1] >= 3:
            print(getFirstSentence(word[0], content))
        else:
            break
    
        
    

