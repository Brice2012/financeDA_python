# ##################################################
# financeda package 2.0
# Financial Data Analysis
# 金融数据分析
# Author: YeJunjie (Brice)
# E_Mail: ye@okwords.cn
# Date: 2025-12-20
# ####################################################

### 完整的单只股票历史数据分析过程
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import scipy.stats as scs
import statsmodels.api as sm
from .functions_stat import stat_describe, stat_norm_tests

plt.rcParams["font.family"] = ["SimHei"]  # 多字体兜底，适配不同系统
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

def stock_diff(data, png_file='stock_diff.png'):
    ## 绘制股票的价格差折线图
    # 增加买入和卖出信号(Diff, Signal)
    # 取最新的21天的数据做分析
    if len(data) > 21:
        data = data.copy()[-21:]
    ## 计算交易信号Signal数据
    #用.diff()方法来计算每日股价变化情况
    data['Diff'] = data['Close'].diff()
    #创建交易信号字段，命名为Signal
    #简单交易策略
    #·当日股价下跌，下一个交易日买入
    #·当日股价上涨，下一个交易日卖出
    #交易信号字段：Signal, diff > 0 Signal=1 卖出，否则Signal=0
    data['Signal'] = np.where(data['Diff'] > 0, 1, 0)

    # 绘制收盘价折线图和交易信号标志
    plt.figure(figsize=(10, 5))
    # 折线图绘制日K线
    data['Close'].plot(linewidth=2, color='k', grid=True)
    # 卖出标志 x轴日期，y轴数值 卖出信号，倒三角
    # matplotlib.pyplot.scatter(x, y, marker, size, color)
    plt.scatter(data['Close'].loc[data.Signal == 1].index,
            data['Close'][data.Signal == 1],
            marker = 'v', s=80, c='g')
    # 买入标志 正三角
    plt.scatter(data['Close'].loc[data.Signal == 0].index,
            data['Close'][data.Signal == 0],
            marker='^', s=80, c='r')
    plt.savefig(png_file)
    plt.show()
    return png_file
    
# 单只股票数据时间序列分析
def stock_tsa(data):
    
    ## 读取数据并进行预处理
    data.dropna(inplace=True) #去掉缺失值, inplace=True表示在原数据上修改
    
    ## 时间序列分析（股票数据，对数收益率, 移动历史波动率, 42D与252D移动平均等数据）
    print('\n股票的42D与252D移动平均数据:')
    print(data[['Close','42d','252d']].tail())
    data[['Close','42d','252d']].plot(figsize=(8,5))
    plt.show()

    print('\n股票的对数收益率, 移动历史波动率等数据:')
    print(data[['Close','Close_Open','Log_Returns','Mov_Vol']].tail())
    data[['Close','Mov_Vol','Close_Open','Log_Returns']].plot(subplots=True, style='b',figsize=(8,5))
    plt.show()
    return data

def stock_tests(data):
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
    stat_norm_tests(log_array)
    
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
