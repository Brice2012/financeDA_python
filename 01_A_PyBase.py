# ######################################
# python 金融数据分析
# Author: 叶俊杰
# Date: 2025-11-22
# #######################################

#%%
### ch01. python编程基础
### Blue 2025-11-16
############################################
# 1.1 环境配置与包的安装
# 1.2 变量、注释和基本输入输出
# 1.3 基本数据类型
# 1.4 组合数据类型(元组列表集合与字典)
# 1.5 语法结构(分支循环和函数)
############################################

#%%
### 1.1 环境配置与包的安装
##########################

## python解释器
# 在python官网 [https://python.org](https://python.org/) 下载合适版本的安装包，
# 页面上选 Downloads --> Windows，然后选择合适的版本(如 3.14.0, 3.9.5 )和操作系统(如 windows 64 位)
# 或者直接到私有源 [https://ai.okwords.cn/soft/](https://ai.okwords.cn/soft/) 下载
# 安装时一定要记住勾选pip, add path 和 添加环境变量等选项
# 安装完成后打开 《命令提示符》 窗口
#     可以通过 开始菜单--windows系统--命令提示符 找到，
#     或者Win+R后输入 cmd 直接打开），
#     或者搜索 cmd 找到后双击打开。如果经常使用，
#     经常使用的话，可在任务栏图标上点右键，选《固定到任务栏》。
# 在命令提示符后输入 `python -V` （注意是大写的V）可查看版本号，说明安装成功。

## 交互式编程工具：jupyter
# 安装：pip install jupyterlab -i https://pypi.tuna.tsinghua.edu.cn/simple
# 设置：jupyter notebook password  #在静默状态下输入两次您需要设置的密码
# 运行：jupyter lab  #在命令行输入即可启动jupyter, 注意一定要保持窗口不要关闭
# 使用：在浏览器中输入 [http://localhost:8888/]，输入设置的密码，即可使用jupyter
# 退出：ctrl+c

## 集成开发环境：pycharm 或 Visual Studio Code
# 下载：[https://www.jetbrains.com/pycharm/download/#section=windows] 请下载社区版（免费）
# 下载：[https://code.visualstudio.com/download] 有安装版和解压版（熟手建议使用解压版）
# 使用：双击使用即可，注意他们都有很多很有用的插件，功能也很强大，可通过一些学习视频了解常用功能。

## 多解释器(Anaconda)
# python不同版本之间存在较大差异，如python2.x和3.x之间是无法兼容的，然后各种第三方包互相之间又有各种兼容问题。解决这一问题的思路是构建多个不同的python环境，不同的环境设置不同的python版本，不同的包配置和不同的包的版本。
# 所以一般的python开发人员会自行维护几个不同的常见的解释器（环境）。
# Anaconda是一款自动帮我们配置不同环境的软件，可以去其官方网站（[https://www.anaconda.com/](https://www.anaconda.com/)）下载windows版本的安装包。安装后即可安装提示设置不同的环境（python版本，包和包的版本），anaconda会自动帮您消除包版本之间的冲突。
# 但anaconda的windows版本有一些不够友好的地方，比如其体积会很容易达到20G甚至100G的程度，如果按默认的安装在C:\anaconda3文件夹，会很快将系统盘撑爆。可以人工去C:\anaconda3\envs文件夹下把一些不用的环境文件夹删除。
# 不同版本的python解释器，anaconda提供的各种环境，在vscode和pycharm中都被定义为 解释器路径 ，可以自由切换，pycharm中还可通过菜单进行配置。
# conda env list          # 查看环境列表
# conda activate fin-data # 激活fin-data环境
# conda deactivate        #退出当前环境

## python包管理工具：pip和uv
# 安装：python -m pip install --upgrade pip
# 查看：pip list

## 课程将用到的包：
# 科学计算与数据处理: 
#    numpy(数组),
#    pandas(数据框),
#    matplotlib(数据可视化),
#    mplfinance(金融图表),
#    scipy(科学计算)
# 金融数据库: 
#    pandas_datareader(经济与金融数据/国外),
#    yfinance(雅虎财经/国外),
#    tushare(金融数据平台/国内),
#    jqdatasdk(聚宽量化平台/国内/付费)
# 统计模型与机器学习: 
#    scikit-learn(机器学习),
#    statsmodels(统计模型),
#    factor-analyzer(因子分析)

## 安装需要的包
# pip install numpy pandas matplotlib mplfinance scipy
# pip install pandas_datareader yfinance tushare jqdatasdk
# pip install scikit-learn statsmodels factor-analyzer

# # 为了提高安装速度，可在每条命令后添加 -i 以使用国内的pip源。如：
# pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
# #如果执行yfinance时报错(python3.9.5时)，可能需要调整urllib3包的版本
# pip install urllib3==1.25.11 

print("需要的包已经安装完成")

## 课程未用到但未来可能会用的包：
# 金融数据：baostock(数据),,quandl(数据源),quantlib(模型),zipline(量化),backtrader(量化),pyfolio(风险),pyalgotrade(工具),alphalens(机器学习),keras-rl(深度学习),ta-lib(指标)
# 数据处理：nltk,spacy,(自然语言处理);opencv,(图像处理);requests,beautifulsoup4,(爬虫);sqlalchemy,pymysql,pymongo,(数据库);
# 数据可视化: seaborn,plotly,pyecharts
# 深度学习：tensorflow,keras,pytorch

#%%
### 1.2 注释、变量和基本输入输出
################################

## python注释符号:
'''
单行：#
多行：三个双引号，或者三个单引号
'''

## hello world
print("hello world!!!")

## 变量：
name="Wang"
age=21
# print("请在输入框中输入一个成绩")
# score=input("请输入你的成绩")
score=90.5

## 格式化输出的四种方式：
print("\n")
print("我的名字叫%s, 年龄为%d岁,\t成绩为%.2f分。" %(name,age,float(score)))
print('我的名字叫{}, 年龄为{}岁,\t成绩为{}分。'.format(name,age,score))
print("我的名字叫" + name + ", 年龄为" + str(age) + "岁。" + "\t成绩为" + str(score) + "分。")

# 格式字符串
# f-string格式化字符串，python3.6及以上版本支持
print(f'我的名字叫{name}, 年龄为{age}岁,\t成绩为{score}分。')

#%%
### 1.3 基本数据类型
#########################

"""
数值型: 
    整型 int: 整数, 可以是正数、负数或零, 没有大小限制(受内存约束)
    浮点型 float: 浮点型表示包含小数部分的实数, 最多取15-17位小数
    复数 complex: 复数类型由实部和虚部构成, 虚部以j或J标记
    布尔型 bool: 布尔型是整型的子类型, 只有True和False两个取值, 对应整数值1和0
字符型: str  ; 
    字符串可以使用三种引号定义，单引号, 双引号, 三引号(用于定义多行字符串), 
    注意py中没有单独的char类型,字符就是1位的字符串
日期时间型: 主要通过time和datetime两个包来处理
    date  2021-5-29
    time  20:19:01
    datetime  2021-5-29 20:19:01
空值: None
"""

print("基本数据类型（数值、字符串、日期时间）\n")

# Python 无需显式定义变量类型，这是由其‌动态类型‌特性决定的。
# 在Python中，变量在第一次被赋值时自动声明，其类型在运行时由解释器根据右侧操作数的值来确定。

a = 3.14159  #定义变量a，赋值为3.14
print(type(a),'\n') #type函数返回变量的类型
b = str(a)
c = "I love python."
d = True
e = 3 + 4j
print(f'变量a的值为{a}，类型为{type(a)}')
print(f'变量b的值为{b}，类型为{type(b)}')
print(f'变量c的值为{c}，类型为{type(c)}')
print(f'变量c的值为{d}，类型为{type(d)}, 转换为数值是{int(d)}')
# 隐式转化的例子：
print(f'隐式转化的例子: a+d+e.real+e.imag的总和为{a+d+e.real+e.imag}')

#%%
## 字符串数据类型基本操作示例

# 特殊符号: \n换行, \t Tab位, \\,
# 格式字符串以"f"开头, 使用花括号{}中的变量填充。
# 原始字符串以"r"开头，不会把反斜线当成特殊字符。
# 多个字符串可使用"+"号拼接。

path=r'G:\publish\codes\02\2.4'
print(f'原始字符串path的内容为:{path}')

str1 = "金融大数据分析课程的成绩是"
score = 90
str_all = str1+str(score)

# 字符串函数(查找, 替换, 切片等)
print(f'\n拼接后的字符串内容为:{str_all}')
print(f'字符串str_all中"课程"两个字的位置为:{str1.find("课程")}')
print(f'字符串str_all的长度为:{len(str_all)}')
print(f'将成绩替换为得分:{str_all.replace("成绩","得分")}')

str3='This is my first book.'
str_list=str3.split(' ')
print(f'\n将字符串{str3}按空格切片的结果为:')
print(str_list)

#%%
## 日期型数据的处理（time包）

import time
d_time=time.time()
print('当前时间的时间戳:', d_time) #返回当前时间的时间戳
l_time=time.localtime() #是一个9维的元组(年份，月份，天，小时，分钟，秒，星期几，一年的第几天，夏令时)
print('当前时间的结构化时间:', l_time) #返回当前时间的结构化时间
print('年份:', l_time[0]) #年份

# strftime(format, time) #将一个时间戳转换为指定格式的字符串
print('\n将时间戳格式化为字符串(strftime): ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) ## 格式化成2016-03-20 11:45:39形式
yesterday_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()-24*3600))
print('昨天: ',yesterday_str)
# strptime(date_string, format) #将一个字符串转换为时间戳
day_time = time.strptime("2021-08-01 09:21:25","%Y-%m-%d %H:%M:%S")
print('将时间字符串转换为时间戳(strptime): ', day_time)

#%%
## 日期型数据的处理（datetime包）

# 与time包相比，datetime包更常用, 更容易理解
from datetime import datetime, timedelta
print('datetime.now():')
print('当前时间: ', datetime.now())
print('年份 year: ',datetime.now().year)
print('月份 month: ', datetime.now().month)
print('日期 day: ',datetime.now().day)
print('小时 hour: ',datetime.now().hour)
print('分钟 minute: ', datetime.now().minute)
print('秒 second: ',datetime.now().second)
print('星期 weekday: ', datetime.now().weekday())
print('时间戳 timestamp: ', datetime.now().timestamp())
print('日期 date: ', datetime.now().date())
print('时间 time: ', datetime.now().time())

yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S %A") # %A 表示星期几
print('\n使用timedelta来获取昨天并格式化为字符串:')
print(type(yesterday))
print(yesterday)

print('\n生成金融中常用的start_date和end_date的两种方法:')
# 最新的接口都已经支持直接使用日期字符串来表达金融
start_date = datetime(2021,8,1) #直接使用datetime函数
#strptime(date_string, format)
end_date = datetime.strptime("2024-12-02 09:21:25","%Y-%m-%d %H:%M:%S") #使用strptime将字符串转换为时间戳
print('start_data', start_date)
print('end_date', end_date)

#%%
### 1.4 组合数据类型(元组, 列表, 集合, 字典)
###########################################

"""
元组, tuple  小括号, 元素固定不变, 有序, 数据类型可不同
列表, list   中括号, 元素可变, 有序, 数据类型可不同
集合, set    大括号, 元素的数据类型相同, 无序
字典, dict   大括号, 元素是 key:value, 无序

嵌套和多维: 各组合类型均支持多重嵌套和相互嵌套。

引用方法 [0]  ,[0:2],  [-1],  ["key"], [0][0]
注意: python index编号是从0开始。

"""

print("组合数据类型（元组、列表、字典、集合）")


#%%
## 元组
a = (1901,"tom","男",40)
print(f'元组a的数据类型位{type(a)}, 值为{a}')
print(f'元组a的-1和0:2表示的元素分别为:')
print(a[-1])
print(a[0:2])

b=((1901,"tom","男",40),(1902,"tom","男",40),(1903,"tom","男",40))
print(f'\n二维元组b的第3个元素的值为{b[2]}')

data_tuple = ("zhangsan","lisi","wangwu")
print(f'\n元组data_tuple的第1个元素的值为:{data_tuple[0]}')
print('元组data_tuple的各元素值依次如下:')
for i in range(len(data_tuple)):
    print(data_tuple[i])

#%%
## 列表
# 列表与元组一样，有序, 支持多种数据类型
# 列表的引用与元组一样，这里就不赘述。
# 不同的是列表支持数据的更新操作(append,insert,extend, remove, pop, del)

# 二维列表示例(类似数组)
a=[[1,2,4],[3,4,5],[5,6,7]]
print(f'二维列表a的数据类型位{type(a)}, \n其值为: {a}')
print(f'二维列表的第2行数据为:{a[1]}')
print(f'二维列表a的第一行第二列的数字为: {a[0][1]}')

# 由字典作为元素组成列表示例（类似数据框）
b = [
    { "name":"ye","sex":"男","age":26},
    { "name":"李四","sex":"男","age":40}
    ]
print(f'\n列表b的数据类型为:{type(b)}, \n列表b的第一个元素b[0]的数据类型为:{type(b[0])}')
print(f'字典b[1]的name值为:{b[1]["name"]}, 其数据类型为:{type(b[1]["name"])}')

# 列表的更新（与元组的区别所在）
# list数据更新有四种方法（append, extend, insert，+）
data_list = ["zhangsan","lisi","wangwu"]
print(f'\n列表更新示例\n列表data_list的初始值为:{data_list}')
data_list.append("yexiaoqi")  # 在列表尾部添加一个元素
data_list.insert(1,"wangwu")  # 在指定索引位置插入一个元素, 有序结构，值可以重复
data_list_add=["fangfang","zhouzhou"]    
data_list.extend(data_list_add)     # 将两个列表合并
print('列表data_list更新后的各元素值依次如下:')
for i in range(len(data_list)):
    print(data_list[i])
    
print(f'\n列表data_list再加上data_list_add之后的新列表为:')
data_all = data_list+data_list_add
print(data_all)

#%%
# 删除元素的方法(del, pop(), remove(), clear())

# Python列表提供了多种删除元素的方法，每种方法适用于不同的场景和需求。
# 根据索引删除元素
# 使用del语句可以根据索引删除列表中的特定元素，这种方法直接删除指定位置的元素，不返回被删除的元素值
del data_all[1]     # 删除索引为1的元素 wangwu
del data_all[4:6]   #会删除索引4到6（不包括6）的元素 fangfang zhouzhou
# 根据值删除元素
data_all.remove("lisi") #方法用于移除列表中第一个匹配指定值的元素，lisi
# 与del不同，remove是根据元素值而不是索引来删除的
# 弹出并返回元素
name_pop = data_all.pop()  #删除列表末尾的元素，并且返回被删除的值供后续使用。通过在括号中指定索引，pop()也可以删除列表中任何位置的元素 zhouzhou
print('清理后的列表\n', data_all)

#%%
# 清空整个列表
data_all.clear() #方法能够一次性清空列表中的所有元素，将列表变为空列表,这种方法没有参数，也没有返回值，直接修改原列表
print('清空后的列表 \n', data_all)
del data_all # 删除整个列表对象不同，clear只是清空列表内容
# print(data_all)  #因为已经删除该变量，print会报错

#%%
## 集合
data_set = {"zhangsan","lisi","wangwu"}
data_set.add("yexiaoqi")
data_set.add("wangwu") #集合为无序结构，重复值被合并
print(f'\n集合data_set最终的内容为:')
for d_set in data_set:
    print(d_set)

#%%
## 字典
# key:value, 无序，依靠key来作为索引。
# key为字符串，且唯一
# value可为python支持的各种数据类型(包括复杂的嵌套结构)
a = {
        "name":"ye",
        "sex":"男",
        "age":26,
        "居住过的城市":["安庆","蚌埠","湘潭","西安","合肥","杭州","苏州","南京","广州","深圳"]
    }
print(f'字典a的数据类型为:{type(a)}\n其初始值为{a}')
print(f'天空没有留下翅膀的痕迹, 可我已经飞过这么多城市{a["居住过的城市"]}。')

# 字典数据及其常见操作
data_dict = {"name":"rose","age":27,"job":"driver"}
print(f"\n字典data_dict初始值为:{data_dict}")
print(f"字典data_dict的age为:{data_dict["age"]}")
print('字典data_dict的keys为:')
print(data_dict.keys())
print('字典data_dict的values为:')
print(data_dict.values())
print('字典data_dict的第1个Value为:')
#注意需要将object对象转换为List才能使用数字索引。
print(list(data_dict.values())[0]) 

print('\n字典data_dict的key循环为:')
for val in data_dict.values():
    print(val)
    
print('\n字典data_dict的key-value循环为:')
for key, value in data_dict.items():
    print(" \" %s \" : \" %s \" " % (key, value))

# 更新字典(update函数)
# update函数按key作为索引来处理数据，如果key不存在就新增，如果key已经存在就替换原value。
data_dict.update({"民族":"汉族","age":101})
print('\n字典data_dict更新后的数据为:')
print(data_dict)

#%%
### 1.5 语法结构(分支循环和函数)
##############################

print("\n条件分支示例,请输入数学和语文成绩的分数(100分制)")
#score1 = int(input("数学分数："))
#score2 = int(input("语文分数："))
score1=90
score2=99
if score1 >= 90:
    if score2 >= 90:
        print("优秀")
    else:
        print("良好")
else:
    if score2 >= 90:
        print("良好")
    else:
        print("不合格")

print("\n循环示例")
names = ['Abe', 'Tom', 'Lily']
for name in names:
    print(name)

print('\n计算10的阶乘')
sum = 0
for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    sum = sum + x
print(sum)

print('\n输出10以内奇数')
n = 0
while n < 10:   
    n = n + 1   
    if n % 2 == 0:  # 如果n是偶数，执行continue语句
        continue # continue语句会直接继续下一轮循环，后续的print()语句不会执行
    print(n)

#%%
## 函数
'''
def 函数名(参数列表):
    //实现特定功能的多行代码
    [return [返回值]]
'''

print("\n函数示例: 根据年龄判断是否成年")
def judge_person(age):
    if age < 18:
        #print("teenager")
        return "teenager"
    else:
        #print("adult")
        return "adult"
result = judge_person(22)
print(result)

#%%
### END ###
# %%
