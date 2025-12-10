# ######################################
# python 金融数据分析
# Author: 叶俊杰
# Date: 2025-11-22
# #######################################

#%%
### ch2. Python数据分析基础
#############################################
# 2.1 数组（numpy包）
# 2.2 数据框（pandas包）
# 2.3 通过pandas读写csv文件并处理金融数据
# 2.4 数据可视化(matplotlib包与mplfinance包)
##############################################

# # python数据分析有两个主要数据结构：numpy数组和pandas数据框
# numpy数组：基本数据结构，可以进行基本的数据操作，与列表的区别在于(要求全部元素的数据类型一致，有更多的矩阵计算相关的函数功能)
# pandas数据框：基于numpy数组的数据结构，提供了更多的数据操作功能

# # 通过Pandas可以读写多种数据来源的数据（囿于时间限制，这里仅介绍csv文件的读写）：
# 1. csv文件（.csv格式, 可用Excelc处理的一种逗号分隔的文本文件）
# 2. Excel文件（.xls或.xlsx格式）
# 3. 数据库文件（如SQLite3, MySQL, MongoDB, PostgreSQL等）
# 4. 网络数据（如HTTP请求、API调用等）
# 5. 其他格式（如JSON、XML等）

# 数据可视化主要时通过matplotlib包与mplfinance包来完成, 专业人士也可以使用eaborn,plotly,pyecharts等包
# matplotlib主要用来汇总折线图、散点图等
# mplfinance 可绘制蜡烛图
#     官网 https://pypi.org/project/mplfinance/
#     示例 https://github.com/matplotlib/mplfinance/blob/master/examples/plotting.ipynb
#     文档 https://mplfinance.readthedocs.io/en/latest/

#%%
## 导入需要的包
# pip install numpy pandas matplotlib mplfinance scipy #使用pip安装需要的包
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import mplfinance as mpf

# %%
### 2.1 数组（numpy包）
#######################

import numpy as np

## 创建一维数组(np.array)并查看数组属性
a = np.array([1,2,3,4,5])
print('数组a的数据: ')
print('数组a: ', a)
print('数组a的第1:2个元素: ',a[0:1])

print('\n数组a的属性: 维度、形状、大小、类型、占用字节等')
print('维度: ', a.ndim)
print('形状: ',a.shape)
print('元素个数: ',a.size)
print('元素数据类型: ',a.dtype)
print('元素占用的字节大小: ',a.itemsize)

## 创建多维数组(np.array)并读取内容
print('\n二维数组b: ')
data1 = [[1, 2, 3, 4, 5], [ 6, 7, 8, 9, 10], [ 7, 7, 7, 0, 3]]
b = np.array(data1)
print('二维数组b的第1行: ',b[1,:])
print('二维数组b的第1行第2列', data1[1][2])

#%%
## 创建特定数组(np.arange, np.zeros, np.ones)

arr3 = np.arange(0,20,2,dtype=float).reshape(2,5)
print('二维数组arr: 2行5列, 0-20步长为2, float类型')
print(arr3)

zes = np.zeros(25,dtype=int).reshape(5,5)
print('\n二维数组zes: 5行5列, zeros数组')
print(zes)
ons= np.ones(9,dtype=int).reshape(3,3)
print('\n二维数组ones: 3行3列, ones数组')
print(ons)

#%%
## 创建随机数组(np.random)
d=np.random.randn(16).reshape(4,4)
print('二维数组d: 4行4列, 随机float类型')
print(d)
print('\n获取数组d的第一行元素: ')
print(d[0,1:-1])
print('\n获取数组d的第一列元素: ')
print(d[:,0])
print('\n数组d取两位小数: ')
print(d.round(2))

#%%
### 2.2 数据框(pandas包)
########################

# pandas是基于numpy的数据分析工具，提供了大量的数据操作功能,

import numpy as np
import pandas as pd

#%%
## 通过数组创建数据框
a = np.random.standard_normal((5,4))  #生成一个随机正态分布的2行3列数组
print('生成5行4列的随机二维数组: ')
print(a)
#print(a[0][2])

# 把数组转换数据框
df = pd.DataFrame(a)
print('\n将二维数组转换为数据框df: ')
print(df)

#%%
## 给数据库加上索引

# 复制数据框
print('\n复制copy()数据框df2:')
df2 = df.copy()
print(df2)

# 给数据框加上行索引(index)
print('\n给数据框加上行索引(index)')
dates =pd.date_range('2025-1-1',periods=5, freq='ME')
df2.index=dates
print(df2.head(5))

# 给数据框加上列索引(columns)
print('\n给数据框加上列索引(columns)')
df2.columns=[['Open', 'Close', 'Volume', 'Price']]
print(df2)

#%%
## 通过字典创建 DataFrame
# 各列数据类型相同，列间数据类型不同的适合使用列表或字典来保存原始数据并导入数据框
print('\n通过字典创建 DataFrame')
df_dict = pd.DataFrame({'name': ["zs", "ww", "liuliu"], 'age': [21, 18, 20]})
print(df_dict)

#%%
# 从 Series 创建 DataFrame
print('\n从 Series 创建 DataFrame')
s1 = pd.Series(['Alice', 'Bob', 'Charlie'])
s2 = pd.Series([25, 30, 35])
s3 = pd.Series(['New York', 'Los Angeles', 'Chicago'])
df_s = pd.DataFrame({'Name': s1, 'Age': s2, 'City': s3})
print(df_s)

#%%
## 数据框的基本操作（索引取值）
print('取 column 为 Close 的列数据: ')
print(df2["Close"])    #取列
print('\n取 column 为 Close 的列的第1个数据: ')
print(df2['Close'].iloc[0])

print('\n取第0行(iloc)数据: ')
print(df2.iloc[0])  #取行

print('\n取第1行第0列(iloc)数据: ')
print(df2.iloc[1,0])

print('\n取 index为2024-01-31 的行(loc)数据: ')
print(df2.loc['2025-01-31'])

#%%
# DataFrame 的属性和方法
print('DataFrame 的属性和方法: ')
print('形状 shape: ', df2.shape)     # 形状
print('列名 columns: ', df2.columns)   # 列名
print('索引 index: ', df2.index)     # 索引
print('\n前5列 head: \n', df2.head())    # 前几行数据，默认是前 5 行
print('\n后3列 tail: \n', df2.tail(3))    # 后几行数据，默认是后 5 行
print('\n描述统计 describe: \n', df2.describe())# 描述统计信息
print('\n均值 mean: \n', df2.mean())    # 求平均值
print('\n求和 sum: \n', df2.sum())     # 求和

#%%
## 绘制累计求和数值的折线图
print('数据框df2的累计求和数值的折线图:')
print(df.cumsum())
df.cumsum().plot(lw=2.0)


#%%
### 2.3 通过pandas读写csv文件并处理金融数据
##########################################

import pandas as pd

## 将数据存入本地excel文件
outputfile = './data/df2_test.csv'
df2.to_csv(outputfile)

# 使用pandas的read_csv()函数读csv数据
# csv是一种使用逗号和分行来分隔的纯文本形式的数据文件，可以使用excel读取
# Excel的数据和数据库的表都可以快速的转换为csv(逗号分隔)格式并保存。
data_df2 = pd.read_csv("./data/df2_test.csv", encoding = "gbk") 
# encoding参数表示文件的编码类型，设为gbk或utf-8才能支持汉字，如果数据中没有中文，可省略这个参数。
print(data_df2)

#%%
## 处理真实的金融数据
import pandas as pd
stock_data = pd.read_csv("./data/stockdata_BABA.csv") 
# 设置列名
stock_data.columns = ['Date','Close','High','Low','Open','Volume']
print(stock_data.head(3))

#%%
## 数据框数据清洗与保存到csv

# 为了数据安全，清洗数据前一般先copy一份。
stock_data2=stock_data.copy() 
# 进行数据清洗（各种处理）
del stock_data2['Volume'] #删除volume列
stock_data2['closeopen'] = (stock_data2['Close']-stock_data2['Open']) #计算close-open
stock_data2.drop(stock_data.columns[[2,3]], axis=1, inplace=True) # 删除high和low列axis=1表示对列进行操作
stock_data2.drop(stock_data.index[[1,2]], axis=0, inplace=True) ## 删除第2和第3行
stock_data2.reset_index(drop=True)
stock_data2.fillna(method='bfill', inplace=True)  #缺失值处理

print('清洗后的数据: ')
print(stock_data2.head())
outputfile = './data/stockdata_BABA_2.csv'
stock_data2.to_csv(outputfile) #保存清洗后的数据
print('清洗前的原数据(可进行比较):')
print(stock_data.head())

#%%
### 2.4 数据可视化(matplotlib包与mplfinance包)
############################################
# 子图(plt.subplot)
    # 折线图(plt.plot)
    # 盒型图(plt.bar)
    # 直方图(plt.hist)
    # 散点图(plt.scatter)
# 符合正态分布的随机模拟数据的散点图（大样本）
# 股票趋势图
    # 单支股票
    # 多支股票
# 蜡烛图(mplfinance.plot)

# 导入需要的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import mplfinance as mpf

#%%
# 子图(plt.subplot)
# 通过 subplot() 函数，你可以在同一图中绘制不同的东西。
# 建立 subplot 网格，高为 2，宽为 2  

# 折线图(plt.plot)
# 折线图是一种展示数据变化趋势的图形，通过连接数据点来展示数据的变化情况。
# 激活第1个 subplot
plt.subplot(2,  2,  1)  
# 数据准备
data=[[1,24],[2,47],[3,8],[4,18]]
dataMat = np.array(data)
X = dataMat[:,0:1]   # 变量x
y = dataMat[:,1]   #变量y
# 绘制散点图 参数：x横轴 y纵轴
plt.plot(X, y, marker='*',)  #绘制折线
plt.plot(X, y, 'r2')  #绘制散点，r-b,g,y,,o-*,1,2
plt.title('折线图')  #指定标题
plt.xlim(0,5)  #指定横坐标取值范围
plt.ylim(0,60)  #指定Y的取值范围
plt.xlabel('X Label')
plt.ylabel('Y')
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
#plt.show()

#盒型图(plt.bar)
# 用于展示数据的离散分布，特别适合用于比较两组数据的分布情况。
# 激活第2个 subplot
plt.subplot(2,  2,  2)  
# 数据准备
x =  [5,8,10] 
y =  [12,30,6] 
x2 =  [6,9,11] 
y2 =  [6,15,7] 
plt.bar(x, y, align =  'center')  #align 对齐 
plt.bar(x2, y2, color =  'g', align =  'center')
plt.title('盒型图') 
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.ylabel('Y drtt') 
plt.xlabel('X axis') 
#plt.show()

# 直方图(plt.hist)
# 直方图是一种对数据分布情况的图形表示，通过图形的高度来显示数据的分布密度。
# 激活第3个 subplot
plt.subplot(2,  2,  3)  
# 数据准备
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
hist,bins = np.histogram(a,bins =  [0,20,40,60,80,100])  
# print (hist) 
# print (bins)
plt.hist(a, bins =  [0,20,40,60,80,100]) 
plt.title("histogram") 
#plt.show()

# 散点图(plt.scatter)
# 散点图是一种展示两组数据之间关系的图形，通过在坐标系中绘制点来展示数据之间的关系。
# 激活第4个 subplot
plt.subplot(2,  2,  4)  
# 数据准备
data=[
    [0.067732,3.176513],[0.427810,3.816464],[0.995731,4.550095],[0.738336,4.256571],[0.981083,4.560815],
    [0.526171,3.929515],[0.378887,3.526170],[0.033859,3.156393],[0.132791,3.110301],[0.138306,3.149813],
    [0.247809,3.476346],[0.648270,4.119688],[0.731209,4.282233],[0.236833,3.486582],[0.969788,4.655492],
    [0.607492,3.965162],[0.358622,3.514900],[0.147846,3.125947],[0.637820,4.094115],[0.230372,3.476039],
    [0.070237,3.210610],[0.067154,3.190612],[0.925577,4.631504],[0.717733,4.295890],[0.015371,3.085028],
    [0.335070,3.448080],[0.040486,3.167440],[0.212575,3.364266],[0.617218,3.993482],[0.541196,3.891471]
]
dataMat = np.array(data)
X = dataMat[:,0:1]   # 变量x
y = dataMat[:,1]   #变量y
# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='*',)
plt.title('scatter plot')
# 显示图形
#plt.show()

plt.show()

#%%
## 符合正态分布的随机模拟数据的散点图（大样本）

y = np.random.standard_normal((1000, 2))
c = np.random.randint(0, 10, len(y))
plt.figure(figsize=(7, 5))
plt.scatter(y[:, 0], y[:, 1], c=c, marker='o')
plt.colorbar()
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')

#%%
# 使用数据框数据绘制股票趋势图
import pandas as pd
stock_data_3 = pd.read_csv("./data/stockdata_BABA.csv") 
# 设置列名
stock_data_3.columns = ['Date','Close','High','Low','Open','Volume']

plt.plot(stock_data_3['Close'], label = f'阿里巴巴股票价格')  #使用数据绘图
plt.xticks(rotation=270)
plt.grid(True)
plt.title('BABA stock')
plt.rcParams['font.sans-serif'] = 'SimHei' #显示中文不乱码
plt.rcParams['axes.unicode_minus'] = False #显示负数不乱码
plt.legend()
plt.show()

#%%
## 股票趋势图(多支股票)
# 通过matplotlib包，我们可以绘制股票的走势图，展示股票的价格变化情况。

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

# 股票代码的写法: 'GOOG','IBM','600030.ss','300481.sz'
# for stock in ['GOOG','AAPL','BABA', 'BIDU']:
#     stock_data = yf.download(stock,'2024-11-1','2025-11-11')
#     outputfile = './data/stockdata_'+stock+'_20251111.csv'
#     stock_data.to_csv(outputfile, header=False) #保存下载的数据
#     plt.plot(stock_data['Close'], label = '{} 股票价格'.format(stock))
for stock in ['GOOG','AAPL','BABA', 'BIDU']:
    inputfile = './data/stockdata_'+stock+'_20251111.csv'
    stock_data =  pd.read_csv(inputfile, encoding = "gbk") 
    stock_data.columns = ['Date','Close','High','Low','Open','Volume']
    plt.plot(stock_data['Close'], label = '{} 股票价格'.format(stock))
plt.xticks(rotation=270)
plt.grid(True)
plt.title('股票走势图')
plt.legend()
plt.rcParams['font.sans-serif'] = 'SimHei' #显示中文不乱码
plt.rcParams['axes.unicode_minus'] = False #显示负数不乱码
plt.show()

#%%
## 使用mplfinance绘制蜡烛图
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

data3 = pd.read_csv("./data/stockdata_BABA.csv", encoding = "gbk") #从csv读取数据
# 设置列名
data3.columns = ['Date','Close','High','Low','Open','Volume'] 
print(data3.tail(5))
data3_2 = data3.copy()[-30:]
# print(data3_2)

# 转换为mplfinance需要的格式
data3_2['Date'] = pd.to_datetime(data3_2['Date'])
data3_2.set_index('Date', inplace=True)

# 绘制蜡烛图
# type='candle', type='line', type='renko', or type='pnf'
mpf.plot(data3_2, type='candle', style='charles', title='BABA Candlestick Chart', ylabel='Price')
# mpf.plot(data3_2, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', savefig='candlestick_chart.png')
plt.show()
# 保存到图片到本地，也可以直接在plot函数中添加savefig参数，但就不会在交互窗口中show()。
plt.savefig('./data/candlestick_chart_2.png')

#%%
# 使用mplfinance绘制不同类型的股票图表
# OHLC 图（开盘价、最高价、最低价、收盘价图）
mpf.plot(data3_2, type='ohlc', style='yahoo', title='OHLC Chart', ylabel='Price')

#%%
# 指定时间
mpf.plot(data3_2['2025-10-15':'2025-10-31'], type='candle', style='yahoo', title='Candlestick Chart (200508)', ylabel='Price')


# %%
##################END##################