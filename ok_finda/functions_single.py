
#%%
### 完整的单只股票历史数据分析过程
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import scipy.stats as scs
import statsmodels.api as sm
from .functions_stat import stat_describe, normality_tests

# 单只股票数据时间序列分析
def stock_tsa(data):
    ## 读取数据并进行预处理
    data.dropna(inplace=True) #去掉缺失值, inplace=True表示在原数据上修改
    data['Date']=pd.to_datetime(data['Date'],format='%Y%m%d')
    data.index = data['Date']
    print(data.tail(3))
    
    ## 时间序列分析（股票数据，对数收益率, 移动历史波动率, 42D与252D移动平均等数据）
    # 计算42d, 252d 移动平均
    data['42d']= data['Close'].rolling(42).mean()
    data['252d'] = data['Close'].rolling(252).mean()
    print('\n股票的42D与252D移动平均数据:')
    print(data[['Close','42d','252d']].tail())
    data[['Close','42d','252d']].plot(figsize=(8,5))
    plt.show()
    # 计算收盘价的对数收益率(Log_Returns), 以及收盘价与开盘价的差值(CloseOpen); shift(1)是向下移动一行, 即当前行数据减去上一行数据
    data['Log_Returns']=np.log(data['Close']/data['Close'].shift(1))
    data['CloseOpen'] = data['Close']-data['Open']
    # 计算移动历史波动率(Mov_Vol: moving annual volatility) 
    data['Mov_Vol'] = (data['Log_Returns'].rolling(252).std())*math.sqrt(252)
    print('\n股票的对数收益率, 移动历史波动率等数据:')
    print(data[['Close','CloseOpen','Log_Returns','Mov_Vol']].tail())
    data[['Close','Mov_Vol','CloseOpen','Log_Returns']].plot(subplots=True, style='b',figsize=(8,5))
    plt.show()
    return data

def stock_candle(data, png_file='stock_candle.png'):
    ## 绘制股票的蜡烛图
    mpf.plot(data, type='candle', style='charles', title='股票蜡烛图', ylabel='价格')
    # 保存图片
    plt.savefig(png_file)
    plt.show()
    return png_file

def stock_test(data):
    ## 正态分布检测（直方图与QQ图等）
    data['Log_Returns']=np.log(data['Close']/data['Close'].shift(1))
    #将对数收益率转换为数组
    log_array = np.array(data['Log_Returns'].dropna())
    # 绘制直方图
    print('\n股票的对数收益率的直方图:')
    data['Log_Returns'].dropna().hist(bins=50)
    plt.show()
    # 输出统计量
    print('\n股票的对数收益率的统计量:')
    stat_describe(log_array)
    # 绘制QQ图
    print('\n股票的对数收益率的QQ图:')
    sm.qqplot(data['Log_Returns'].dropna(), line='s')
    plt.show()
    # 输出偏度、峰度、正态性检验
    normality_tests(log_array)
    
    p_value = scs.normaltest(log_array)[1]
    if p_value <= 0.05:
        print("对数收益率通过正态性检验 (p-value=%.4f <= 0.05)" % p_value)
        return True
    elif p_value <= 0.1:
        print("对数收益率通过正态性检验 (p-value=%.4f <= 0.1)" % p_value)
        return True
    else:
        print("对数收益率未通过正态性检验 (p-value=%.4f > 0.1)" % p_value)
        return False
