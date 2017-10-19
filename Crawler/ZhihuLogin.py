import requests
import re
import time
import os.path
try:
    import cookielib
except:
    import http.cookiejar as cookielib
try:
    from PIL import Image
except:
    pass
    

#header
agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
headers = {
    'Host' : 'www.zhihu.com',
    'Referer' : 'https://www.zhihu.com/',
    'User-Agent' : agent
}

#session对象  
session = requests.session()
#读取之前登录的cookie
session.cookies = cookielib.LWPCookieJar(filename='cookies')
try:
    session.cookies.load(ignore_discard=True)
except:
    print('Cookie未能加载')

#获取动态变化的隐藏参数_xsrf
def get_xsrf():
    url = 'https://www.zhihu.com'
    try:
        page = session.get(url, headers=headers)
    except:
        print('something error')
        return 0
    html = page.text
    pattern = r'name="_xsrf" value="(.*?)"'
    _xsrf = re.findall(pattern, html)
    return _xsrf[0]
    
#获取验证码和验证码要求
def get_captcha():
    t = str(int(time.time()*1000))
    #去除lang=ch，获得英文+数字验证码，data去掉captcha_type字段
    captcha_url = 'https://www.zhihu.com/captcha.gif?r='+t+'&type=login'#&lang=cn'
    r = session.get(captcha_url, headers=headers)
    with open('captcha.jpg','wb') as f:
        f.write(r.content)
    try:
        im = Image.open('captcha.jpg')
        im.show()
    except:
        print('请自行查看验证码')
    captchas = input('input location\n')
    #定位不准确
    '''points = [[12.95,22],[36.1,22],[57.16,22],[84.52,22],[108.72,22],[132.95,22],[151.89,22]]
    s = ''
    for i in captchas:
        s += str(points[int(i)-1]) +','
    print(s)
    s = s.strip(',')'''
    captcha = '{"img_size":[200,44],"input_points":['+ captchas +']}'
    print(captcha)
    return captcha

def login(account, passwd):
    _xsrf = get_xsrf()
    headers['X-Xsrftoken'] = _xsrf
    headers['X-Requested-With'] = 'XMLHttpRequest'
    post_data = {
        '_xsrf' : _xsrf,
        'password' : passwd,
        'captcha_type' : 'cn'
    }
    #根据账号类型进行登录
    if re.match(r'^1\d{10}$', account):
        print('手机号登录')
        post_url = 'https://www.zhihu.com/login/phone_num'
        post_data['phone_num'] = account
    else:
        if '@' in account:
            print('邮箱登录')
        else:
            print('账号有问题')
            return 0
        post_url = 'https://www.zhihu.com/login/email'
        post_data['email'] = account
    try:
        login_page = session.post(post_url, data=post_data, headers=headers)
    except:
        print('something error')
        return 0
    login_code = login_page.json()
    if login_code['r'] == 1:
        post_data['captcha'] = get_captcha()
        login_page = session.post(post_url, data=post_data, headers=headers)
        login_code = login_page.json()
        print(login_code['msg'])
    session.cookies.save()

if __name__ == '__main__':
    account = input('请输入你的用户名\n>  ')
    passwd = input("请输入你的密码\n>  ")
    login(account, passwd)
