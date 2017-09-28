from selenium import webdriver
import time
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import StaleElementReferenceException

def waitForLoad(driver):
    elem = driver.find_element_by_tag_name('html')
    count = 0
    while True:
        count +=1
        if count > 20:
            print('timing out after 10s and returning')
            return
        time.sleep(.5)
        try:
            #单纯比较不会抛出异常，要raise
            if elem != driver.find_element_by_tag_name('html'):
                raise StaleElementReferenceException
        except StaleElementReferenceException:
            print('redirected')
            return

if __name__ == '__main__':
    driver = webdriver.PhantomJS(executable_path='phantomjs\\bin\\phantomjs.exe')
    driver.get('http://pythonscraping.com/pages/javascript/redirectDemo1.html')
    waitForLoad(driver)
    print(driver.page_source)
