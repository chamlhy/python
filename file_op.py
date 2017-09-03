def savefile(boy, girl, count):
    girl_name = 'girl_' + str(count) + '.txt'
    boy_name = 'boy_' + str(count) + '.txt'
    girl_file = open(girl_name,'w')
    boy_file = open(boy_name,'w')
    girl_file.writelines(girl)
    boy_file.writelines(boy)
    girl_file.close()
    boy_file.close()
    

def splitfile():
    f = open('对话.txt')
    boy = []
    girl = []
    count = 1
    for each_line in f:
        if each_line[:6] != '======':
            (role,spoken) = each_line.split(':',1)
            if role == '七巧姐':
                girl.append(spoken)
            else:
                boy.append(spoken)
        else:
            savefile(boy, girl, count)
            boy = []
            girl = []
            count += 1
    savefile(boy, girl, count)

splitfile()
