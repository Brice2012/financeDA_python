# ######################################
# python 金融数据分析
# Author: 叶俊杰
# Date: 2025-11-28
# #######################################

#%%
### ch4. 时间序列分析（单支股票）
#################################
# 4.1 概述
# 4.2 常见分析变量
# 4.3 移动窗口分析
# 4.4 高频数据分析

#%%
# 导入需要的包
# pip install panda numpy matplotlib scipy  #安装数据处理基本包
import datetime, math #导入时间和数学包（python默认已经安装）
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

#%% 
### 4.1 概述
############
'''

金融数据时间序列分析
使用股票（期货，期权）的时间序列数据，可以构建如Diff, Log_Rets, 42d, 252d, Mov_Vol等用于分析的变量。
有两大类，一类是简单分析（包括Diff, Signal, CloseOpen, OpenClose, HighLow, Log_Rets,等），一类是基于移动窗口的分析（包括42d, 252d, Mov_Vol等）。
可以绘制适当的图形来可视化数据可辅助分析

常用的有如下2类5组9个变量，具体计算公式和相关知识点如下：
1. 买入和卖出信号(Diff, Signal)
    data['Diff'] = data['Close'].diff()                 # 收盘价的差分, 即当前行数据减去上一行数据
    data['Signal'] = np.where(data['diff'] > 0, 1, 0)   # 当diff大于0时, 信号为1, 否则为0
2. 价差(CloseOpen, OpenClose, HighLow)
    data['CloseOpen'] = data['Close']-data['Open']  # 收盘价与开盘价的差值
    data['OpenClose'] = data['Open']-data['Close']  # 开盘价与收盘价的差值
    data['HighLow'] = data['High']-data['Low']      # 最高价与最低价的差值
3. 收益率与对数收益率(returns, Log_Rets) 
    收盘价的收益率与对数收益率, shift(1)是向下移动一行, 即当前行数据减去上一行数据
    data['returns'] = data['Close'].pct_change()
    data['Log_Rets'] = np.log(data['Close']/data['Close'].shift(1)) 
4. 移动平均(42d, 252d), 移动平滑法
    与移动平均有关的函数应用; rolling(windows=42).mean() ,min(),max(),corr(),std(), 42天和252天的移动平均线, 是收盘价的平均值, 表示短期和长期的趋势
    data['42d']= data['Close'].rolling(42).mean()
    data['252d'] = data['Close'].rolling(252).mean()
5. 移动历史波动率 (Mov_Vol: moving annual volatility)
    moving annual volatility, 是收盘价的对数收益率的年化标准差, 即对数收益率的标准差乘以252的平方根
    data['Mov_Vol'] = (data['Log_Rets'].rolling(252).std())*math.sqrt(252)

高频数据分析 High Frequency Data (BID是买入价, ASK是卖出价)
1. 均值 eur_usd['Mid'] = eur_usd.mean(axis=1)
2. 重新采样 eur_usd_resam = eur_usd.resample(rule='10min').last()  #重新采样，.last()可替换成 .mean()

'''
print("准备好了")

#%%
### 4.2 常见分析变量
#################################
import math, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# 金融数据时间序列分析(（包括Diff, Signal, CloseOpen, OpenClose, HighLow, Log_Rets,等）)

# 读取csv文件
data = pd.read_csv("./data/stockdata_BABA.csv", encoding = "gbk") #从csv读取数据
data.columns = ['Date','Close','High','Low','Open','Volume']
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print(data.head())

#%%
# 增加买入和卖出信号(Diff, Signal)

# 取最新的30天的数据做分析
data3 = data.copy()[-30:]
print(data3)

## 计算交易信号Signal数据
#用.diff()方法来计算每日股价变化情况
data3['Diff'] = data3['Close'].diff()
#创建交易信号字段，命名为Signal
#简单交易策略
#·当日股价下跌，下一个交易日买入
#·当日股价上涨，下一个交易日卖出
#交易信号字段：Signal, diff > 0 Signal=1 卖出，否则Signal=0
data3['Signal'] = np.where(data3['Diff'] > 0, 1, 0)

# 绘制收盘价折线图和交易信号标志
plt.figure(figsize=(10, 5))
# 折线图绘制日K线
data3['Close'].plot(linewidth=2, color='k', grid=True)
# 卖出标志 x轴日期，y轴数值 卖出信号，倒三角
# matplotlib.pyplot.scatter(x, y, marker, size, color)
plt.scatter(data3['Close'].loc[data3.Signal == 1].index,
        data3['Close'][data3.Signal == 1],
        marker = 'v', s=80, c='g')
# 买入标志 正三角
plt.scatter(data3['Close'].loc[data3.Signal == 0].index,
        data3['Close'][data3.Signal == 0],
        marker='^', s=80, c='r')
plt.show()

#%%
# 计算价差 (CloseOpen, OpenClose, HighLow) 和收盘价的对数收益率(Log_Rets); 
# shift(1)是向下移动一行, 即当前行数据减去上一行数据

data1 = data.copy() # 以下使用全量数据
print(data1.head())

data1['Close'].plot(figsize=(8,5))

data1['Diff'] = data1['Close'].diff()
data1['CloseOpen'] = data1['Close']-data1['Open']
data1['OpenClose'] = data1['Open']-data1['Close']
data1['HighLow'] = data1['High']-data1['Low']
data1['returns'] = data1['Close'].pct_change()
data1['Log_Rets']=np.log(data1['Close']/data1['Close'].shift(1))
print(data1[['Close', 'CloseOpen', 'OpenClose', 'HighLow', 'returns', 'Log_Rets']].head())

data1[['Close','CloseOpen','OpenClose','HighLow','returns','Log_Rets']].plot(subplots=True, style='b',figsize=(8,5))
plt.show()

#%%

### 4.3 移动窗口分析
######################

# 移动窗口的概念和移动平滑法是一种常见的时间序列分析方法。
# 与移动平均有关的函数应用，rolling(windows=42).mean(),min(),max(),corr(),std()

#%%
# 计算42d, 252d 移动平均

# data1['42d'] = pd.rolling_mean(data1['Close'],windows=42) #出错，需要用下面这句
data1['42d']= data1['Close'].rolling(42).mean()
data1['252d'] = data1['Close'].rolling(252).mean()
print(data1[['Close','42d','252d']].tail())

data1[['Close','42d','252d']].plot(figsize=(8,5))
plt.show()

#%%
# 计算移动历史波动率(Mov_Vol: moving annual volatility) 

data1['Mov_Vol'] = (data1['Log_Rets'].rolling(252).std())*math.sqrt(252)
print(data1[['Close','Log_Rets','Mov_Vol']].tail())
data1[['Close','Log_Rets','Mov_Vol']].plot(subplots=True, style='b',figsize=(8,5))
plt.show()

#%%
# 将前述计算结果保存到csv
outputfile = './data/stockdata_BABA_TSA.csv'
data1.to_csv(outputfile)

#%% 
# mplfinance高级绘图
import mplfinance as mpf

data2 = data1.copy()[-20:]
print(data2)

#%%
# 绘制蜡烛图
# type='candle', type='line', type='renko', or type='pnf'
mpf.plot(data2, type='candle', style='charles', title='BABA Candlestick Chart', ylabel='Price')
# plt.show()
# 添加技术指标（移动平均线）
add_plot = [
    mpf.make_addplot(data2['42d'], color='b'),
    mpf.make_addplot(data2['252d'], color='r')]
mpf.plot(data2, type='candle', style='yahoo', title='Candlestick Chart with Moving Averages', ylabel='Price', addplot=add_plot)
#plt.show()
#plt.savefig('./data/candlestick_chart.png')

#%%
# 综合分析

from functions_tsa import tsa
file_ = tsa(stock_code='002594.SZ')
print(file_)

#%%
# 4.4 高频数据分析（股票）
########################

# High Frequency Data
# data from FXCM Forex Capital Markets Ltd.
#data = pd.read_csv('http://hilpisch.com/fxcm_eur_usd_tick_data.csv',index_col=0, parse_dates=True)
eur_usd = pd.read_csv('./data/fxcm_eur_usd_tick_data.csv',index_col=0, parse_dates=True)
eur_usd.info()

eur_usd['Mid'] = eur_usd.mean(axis=1)
eur_usd['Mid'].plot(figsize=(10, 6))
plt.show()

#eur_usd_resam = eur_usd.resample(rule='1min', label='last').last()  #重新采样
#eur_usd_resam = eur_usd.resample(rule='10min').last()  #重新采样
#eur_usd_resam = eur_usd.resample(rule='5min',how='mean')
eur_usd_resam = eur_usd.resample(rule='10min').mean()
print(np.round(eur_usd_resam.head(),2))
eur_usd_resam['Mid'].plot(grid=True)
plt.show()

def reversal(x):
    return 2 * 1.16 - x

eur_usd_resam['Mid'].apply(reversal).plot()    
plt.show()

#%%
#############END################"