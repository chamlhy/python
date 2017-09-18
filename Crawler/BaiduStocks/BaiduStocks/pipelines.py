# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


class BaidustocksPipeline(object):
    def process_item(self, item, spider):
        return item

class BaidustocksInfoPipeline(object):
    #爬虫开始时调用
    def open_spider(self, spider):
        self.f = open('BaiduStockInfo.txt', 'w')
	
	#爬虫结束时调用
    def close_spider(self, spider):
        self.f.close()
		
	#处理每一个项目时调用
    def process_item(self, item, spider):
        try:
            line = str(dict(item)) + '\n'
            self.f.write(line)
        except:
            pass
        return item