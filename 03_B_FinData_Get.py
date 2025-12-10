# ######################################
# python 金融数据分析
# Author: 叶俊杰
# Date: 2025-11-28
# #######################################

#%%
### ch3. 金融数据处理
################################################
# 3.1 概述
# 3.2 使用yfinance获取金融数据(yfinance包, 国外)
# 3.3 使用tushare获取金融数据(tushare包, 国内)
# 3.4 使用wb获取经济数据(pandas_datareader包, 国外)
# 3.5 量化平台API接口（jqdatasdk包, 付费）

#%%
# 导入需要的包
# pip install pandas_datareader yfinance tushare jqdatasdk  #安装金融数据源相关的包

import datetime, math #导入时间和数学包（python默认已经安装）
import pandas as pd

import yfinance as yf  # 雅虎财经 金融数据, 国外
import tushare as ts   # TUSHARE数据 金融数据, 国内
import pandas_datareader.wb as wb  # 可用于获取世界银行的经济数据, 国外
import jqdatasdk as jq # 聚宽量化平台 量化平台, 付费

#%% 
### 3.1 概述
###############
'''
获取的数据一般为pandas数据框来处理, 通过df.to_csv()可将数据存入CSV文件中, 供 pd.read_csv() 调用
国外接口需要科学上网。

金融数据在线获取目前较好的方式如下：
1. 通过 yfinance 包 获取股票数据
    Yahoo Finance - Stock Market Live, Quotes, Business.
    官网 https://finance.yahoo.com/
        https://pypi.org/project/yfinance/
2.  通过 Tushare 接口获取(国内接口, 需要注册获取key)
    可获取沪深股市、指数、期货期权、美股、港股等数据
    官网 https://tushare.pro/ (老网站 http://tushare.org/)
        https://tushare.pro/document/2
3.  通过 pandas_datareader 包 获取经济、金融数据
    官网 https://pandas-datareader.readthedocs.io
    Functions from pandas_datareader.data and pandas_datareader.wb extract data from various Internet sources into a pandas DataFrame. 
    股票数据源(API)有: stooq(美国股市),  Naver Finance(韩国股市), 
                    Yahoo Finance(暂不可用,可使用yfinance包替代), 
                    IEX(需要申请API Key), Alpha Vantage(需要申请API Key)
                    quandl(需要申请API Key, 也可以直接使用quandl包https://www.quandl.com/), 
    经济数据源(API)有: World Bank, FRED(Federal Reserve Economic Data), St.Louis FED, Eurostat, OECD, ECB, EIA, etc.
4. 通过量化数据平台, 如聚宽量化 jqdatasdk包 ,可获取除一般的金融数据外的其他经过处理的数据, 如各种因子数据
5. 通过手工方式收集, 录入Excel并整理, 然后另存为csv文件格式, 在使用pandas的pd.read_csv()函数读取为数据框。

'''
print("准备好了")

#%%
### 3.2 使用yfinance获取金融数据(yfinance包, 国外)
#################################################

import yfinance as yf
# 如果报no timezone found错误，可能需要将urllib3包设置为1.25.11版本。因为最新的urllib3包在有代理的情况下不好用。
# pip install urllib3==1.25.11

data = yf.download("BABA", period="10d", auto_adjust=True)   #下载最近10天的数据
# data = yf.download("BABA", period="max", auto_adjust=True) #下载阿里巴巴的全部数据
# data = yf.download("AAPL", start="2024-11-11", end="2025-11-11",  auto_adjust=True) #下载苹果的指定日期数据
data.columns = ['Close','High','Low','Open','Volume']  #修改列名，YF默认得到的是一个复杂格式的列名，这里统一改为Close,High,Low,Open,Volume，索引是Date(日期)
print(data.tail())
print(data.describe())
outputfile = './data/stockdata_BABA_10d.csv'
data.to_csv(outputfile)
# 选择需要的列保存, 如果不保存日期，可指定index=False, 不需要指定 header=False
# data.to_csv(outputfile, header=False) # 如果前面没有修改列名，可以去掉列头保存下载的数据，读取的时候再强制添加列名即可。
# 对应的列分别为 ['Date','Close','High','Low','Open','Volume']

#%%
# 下载中国股市数据

# 股票代码的写法: 'GOOG','IBM','FB','AAPL','BABA','BIDU','600030.ss','300481.sz'
# 港股输入代码+对应股市，如腾讯:0700.hk
# 韩股: 三星:005930.KS
# 上证综指 000001.ss, 深证成指 399001.sz, 沪深300指数代码 000300.ss
data_000300ss = yf.download("000300.ss", period="3d", auto_adjust=True)
print(data_000300ss.head(3))

#%%
# # 下载美股阿里巴巴股票的全部历史数据（已备后用）
data = yf.download("BABA", period="max", auto_adjust=True)
data.columns = ['Close','High','Low','Open','Volume']
outputfile = './data/stockdata_BABA.csv'
data.to_csv(outputfile)

# outputfile 为字符串变量，指定了输出的文件名和路径, ./data 表示在当前目录下的data子目录
# tocsv 函数常见的参数包括：columns, index, header, sep, na_rep, encoding等 
# 如果不保存日期，可指定index=False, 如果不想保存表头可指定 header=False
# 可通过columns指定保存的列
# data.to_csv(outputfile, columns=['Close','High','Low','Open'], index=False, header=False)

#%%
# 获取股票数据（读取已保存的CSV文件）
import pandas as pd

# 假设我们有一个包含日期和股票价格的CSV文件，我们可以使用Pandas库来读取和处理数据。
df = pd.read_csv("./data/stockdata_BABA.csv", encoding = "gbk") #从csv读取数据

# 设置列名（如果使用header=False保存的数据需要手动设置列名，如果使用header=True保存的数据则不需要） 
# df.columns = ['Date','Close','High','Low','Open','Volume']

# 将日期列转换为日期时间类型，并设置为索引(如果不处理成日期型，后续使用mpl绘图时会报错)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print(df.head())
print(df.tail())

#%%
### 3.3 使用tushare获取金融数据(tushare包, 国内)
#################################################
# https://tushare.pro/document/2

"""
重要的事情说三遍: 注意注意注意
一定要注意tushare获取的数据与yfinance获取数据的差异, 
一定要掌握具体的处理方法。
主要差异有如下几点：
1. 数据列的名称不同, 可通过列选择的方式将数据强行整理成我们约定(熟悉)的顺序和名称
    ['Date','Close','High','Low','Open','Volume']
2. 写入csv的参数不同
    因为 yfinance 获取数据的列索引时一个复合索引, 为了处理方to_csv() 时就需要强行指定列名或设置header=False。
    但tushare 的列索引时正常的, 不需要添加header=False, 否则会导致读取的时候少掉第一行数据, 但tushare数据已经建立了1,2,3...的序号索引, 保存时需要自定义索引再保存或者去掉索引保存, 不然会多出一列。
    index=False
3. 日期的格式不同
    雅虎财经的日期数据时标准的年/月/日数据, python中转为datetime类型时直接使用to_datetime()即可
    而tushare的日期数据时年月日组成的字符串, to_datetime()函数需要添加format参数
    df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
5. 数据的顺序不同
    雅虎财经时从旧到新, tushare时从新到旧, 可使用iloc[::-1]将其反转过来。
    data = data.iloc[::-1]
6. 国内股票的名称后缀不同
    雅虎财经是.ss和.sz结尾, tushare是.SH和.SZ结尾, 所以在处理股票代码时需要注意转换。
"""

import tushare as ts # 导入tushare
# 新版本tushare接口使用(最简示例)

## 初始化pro接口
pro = ts.pro_api('your_tushare_token')
# 免费token资源有限
# 请同学们个人去tushare官网免费申请自用的token并替换

#%%
## 获取指定股票指定日期的日线数据（默认最近30天）
# 注意股指代码可能需要额外权限

stock_code = "000630.SZ"
start_date = "20241111"
end_date = "20251111"

stock_data = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date).rename(columns={'trade_date':'Date'})
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y%m%d') # 转换成标准的日期数据格式2025-01-01并设置成索引
stock_data.set_index('Date', inplace=True)
stock_data = stock_data[['close','high','low','open','vol']] # 抽取常见的5列数据
stock_data.columns = ['Close','High','Low','Open','Volume']  #修改列名，方便后续分析（与yf的统一）
stock_data = stock_data.iloc[::-1]  # tushare数据是从新到旧的，需要颠倒过来

print(stock_data.head())
print(stock_data.tail())
print(stock_data.describe())

stock_data.to_csv('./data/stockdata_000630SZ_20241111-20251111.csv') 


#%%
import pandas as pd

df_000630SZ = pd.read_csv("./data/stockdata_000630SZ_20241111-20251111.csv", encoding = "gbk") #从csv读取数据

# 设置列名(如果保存的时候有列名这里可省略)
# df_000630SZ.columns = ['Date','Close','High','Low','Open','Volume']

# 将日期列转换为日期时间类型，并设置为索引
df_000630SZ['Date'] = pd.to_datetime(df_000630SZ['Date'])
# df_000630SZ['Date'] = pd.to_datetime(df_000630SZ['Date'], format="%Y%m%d") # 如果保存的时候未处理这里就要加上格式
df_000630SZ.set_index('Date', inplace=True)

# 显示数据前后5行
print(df_000630SZ.head())
print(df_000630SZ.tail())

#%%
# 日线数据完全版（示例）
# 拉取日线数据(PRO版), 可以设置完整参数，也可以只设置ts_code和日期区间。
df = pro.daily(**{
    "ts_code": '600030.SH',
    "trade_date": "",
    "start_date": "20250101",
    "end_date": "20251130",
    "offset": "",
    "limit": ""
}, fields=[
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount"
])
print(df)
df.to_csv('./data/stockdata_600030SH_2025_ts.csv') # 直接保存
#df.to_csv('./data/stockdata-tushare-600030.SH-2024.csv',columns=['open','high','low','close','vol']) #选择保存

#%%
# 拉取公司基本信息数据
df = pro.stock_company(**{
    "ts_code": "600030.SH",
    "exchange": "",
    "status": "",
    "limit": "",
    "offset": ""
}, fields=[
    "ts_code",
    "exchange",
    "chairman",
    "manager",
    "secretary",
    "reg_capital",
    "setup_date",
    "province",
    "city",
    "website",
    "email",
    "employees",
    "introduction",
    "office",
    "ann_date",
    "business_scope",
    "main_business"
])
print(df)

#%%
### 3.4 使用wb获取经济数据(pandas_datareader包, 国外)
####################################################

# 世界银行经济数据
# pandas users can easily access thousands of panel data series from the World Bank’s World Development Indicators by using the wb I/O functions.

import pandas_datareader.wb as wb  #可用于获取世界银行数据
import pandas as pd

#%%
#获取四国GDP数据并分析
dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX', 'CN'], start=1900, end=2025)
print(dat['NY.GDP.PCAP.KD'].groupby(level=0).mean())
outputfile = './data/WDdata_4g_2025.csv'
dat.to_csv(outputfile) #保存下载的数据

print('四国GDP数据分析')
data5b = pd.read_csv("./data/WDdata_4g_2025.csv",encoding = "gbk")
print(data5b)
data5b=data5b.groupby('country')
print(data5b['NY.GDP.PCAP.KD'].mean())

#%%
#获取2024年各国GDP，投资和汇率数据
#ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']#手机数无法获取
ind = ["NY.GDP.MKTP.CN","NE.GDI.FTOT.CN","PX.REX.REER"]  #从世界银行WDI数据库下载GDP、投资、实际有效汇率数据
dat = wb.download(indicator=ind, country='all', start=2024, end=2024).dropna()
dat.columns = ['gdp', 'tz','hl']
print(dat.tail())
outputfile = './data/WDdata_GDP_TZ_HL_2024.csv'
dat.to_csv(outputfile) #保存下载的数据, 这个数据会用于后续的线性回归分析

#%%
### 3.5 量化平台API接口（jqdatasdk包, 付费）
#############################################

# 现在有很多现成的量化交易平台，不但有大量已经整理好的数据， 并且可以在平台上直接编写策略并进行回测。
# Quantopian 的开源回测工具zipline可以说在业内是无人不知，无人不晓，但zipline 在本地的配置有点麻烦，而Quantopian平台上面的中国股市数据包还需 要单独付费。
# 国内的量化平台也有几个比较有名的，如聚宽（JoinQuant）、米筐 （RiceQuan）、BigQuant等。

#JoinQuant聚宽量化投研平台 [https://www.joinquant.com/]
# pip install jqdatasdk
import jqdatasdk as jq
# jq.auth('账号','密码') #账号是申请时所填写的手机号；密码为聚宽官网登录密码
# jq.get_price(security, start_date=None, end_date=None, frequency='daily', fields=['open','close','low','high','volume','money','factor',
#         'high_limit','low_limit','avg','pre_close','paused'], skip_paused=False, fq='pre', count=None,round=True)
# jq.get_ticks(security, start_dt, end_dt, count, fields, skip=True,df=True)

print('免费使用日期用完了, 没法玩了：）')

#%%
#############END################"