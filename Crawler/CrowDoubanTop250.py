import os
from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv

#豆瓣top250的页面链接
def getTopPages():
    for i in range(10):
        yield 'https://movie.douban.com/top250?start=' + str(25 * i)

#获取页面电影信息并保存图片
def getMovieInfo(infoList, page):
    html = urlopen(page)
    bsObj = BeautifulSoup(html, 'html.parser')
    for item in bsObj.find_all('div', {'class':'item'}):
        img = item.find('img')
        saveImg(img['alt'], img['src'])
        info = []
        info.append(item.find('span', {'class':'title'}).get_text())
        temp = item.find('p').get_text()
        dir_act = temp.split('\n')[1].split(': ')
        info.append(dir_act[1].split('\xa0')[0])
        if len(dir_act) >= 3:
            info.append(dir_act[2])
        elif len(dir_act) == 2:
            info.append('')
        info.append(item.find('span', {'class':'rating_num'}).get_text())
        info.append(item.find('span', {'property':'v:best'}).next_sibling.next_sibling.string.split('人')[0])
        try:
            info.append(item.find('span', {'class':'inq'}).string)
        except:
            info.append('')
        print(info)
        infoList.append(info)

#保存图片
def saveImg(name, imgUrl):
    ext = imgUrl.split('.')[-1]
    downloadDirectory = 'downloadimg\\' + name + '.' + ext
    directory = os.path.dirname(downloadDirectory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    urlretrieve(imgUrl, downloadDirectory)

#保存信息到csv文件
def saveInfo(infoList):
    csvFile = open('info.csv', 'w+', encoding='utf-8')
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('电影名称', '导演', '主演', '评分',  '评分人数', '引述'))
        for info in infoList:
            writer.writerow(info)
    finally:
        csvFile.close()

if __name__ == '__main__':
    infoList = []
    for page in getTopPages():
        getMovieInfo(infoList, page)
    saveInfo(infoList)
