import numpy as np
import pandas as pd
import yfinance as yf
import tushare as ts
import math
from datetime import datetime

# 获取csv格式股票数据（没有数据文件的话自动下载）
def load_stock_csv(stock_code, start_date=None, end_date=None, csv_file=None, token=None, source='tushare'):
    """
    加载股票数据，先尝试从文件加载，若文件不存在则从指定源下载并保存。
    :param stock_code: 股票代码，如 '601318.SH'
    :param start_date: 开始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :param output_file: 输出的pickle文件路径,会自动加上./data/前缀
    :param source: 数据来源，'yf'表示yfinance，'ts'表示tushare，默认'yf'
    :return: 指定时间段内的股票交易数据(DataFrame)
    """
    if start_date is None:
        start_date = '1990-01-01'
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if csv_file in ['', None]:
        csv_file = 'stock_'+stock_code.split('.')[0]+'_'+start_date.replace('-','')+'_'+end_date.replace('-','')+'.csv'
    # 首先让程序尝试读取已下载并保存的文件
    try:
        df = pd.read_csv(csv_file)
        #如果文件已存在，则打印载入股票数据文件完毕
        print(f'成功从文件 {csv_file} 加载股票数据!')
    #如果没有找到文件，则重新进行下载
    except FileNotFoundError:
        print('文件未找到，重新下载中')
        #默认下载源为yahoo，若指定为tushare，则从tushare下载，需要指定token
        if source == 'tushare':
            if token is None:
                print(f'未指定tushare token，无法从tushare下载股票数据 {stock_code}')
                raise ValueError("tushare数据源需要指定token")
            df = get_stock_data_ts(stock_code, start_date, end_date, token=token)
        elif source == 'yfinance':
            df = get_stock_data_yf(stock_code, start_date, end_date)
        else:
            print(f'未知数据源 {source}，无法下载股票数据 {stock_code}')
            raise ValueError("未知数据源，必须为'yfinance'或'tushare'")
        # df = df.set_index('Date')
        #下载成功后保存为pickle文件
        df.to_csv(csv_file)
        #并通知我们下载完成
        print(f'成功下载股票数据 {stock_code} 并保存到文件 {csv_file}')
    #最后将下载的数据表进行返回
    return df

# 使用tushare获取股票数据
def get_stock_data_ts(stock_code, start_date, end_date, token=None):
    """
    使用tushare获取股票数据
    :param stock_code: 股票代码，如 '000001.SZ'
    :param start_date: 开始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :return: 指定时间段内的股票交易数据(DataFrame)
    """
    if token is None:
        print(f'未指定tushare token，无法从tushare下载股票数据 {stock_code}')
        raise ValueError("tushare数据源需要指定token")
    pro = ts.pro_api(token)
    # API-key来自不易，有次数限制，请同学们自己去tushare申请自己的API-key并替换上面的字符串
        # 如果soock_code是以.SZ或.SH结尾，则分别替换成.ss或.sz
    if stock_code.endswith('.sz'):
        stock_code = stock_code.replace('.sz','.SZ')
    elif stock_code.endswith('.ss'):
        stock_code = stock_code.replace('.ss','.SH')
    start_date=start_date.replace('-','')
    end_date=end_date.replace('-','')
    stock_data = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date).rename(columns={'trade_date':'Date'})
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y%m%d') # 转换成标准的日期数据格式2025-01-01
    stock_data.set_index('Date', inplace=True)
    stock_data = stock_data[['close','high','low','open','vol']] # 抽取常见的5列数据
    stock_data.columns = ['Close','High','Low','Open','Volume']  #修改列名，方便后续分析（与yf的统一）
    stock_data = stock_data.iloc[::-1]  # tushare数据是从新到旧的，需要颠倒过来
    return stock_data.dropna()

# 使用yfinance获取股票数据
def get_stock_data_yf(stock_code, start_date, end_date):
    """
    使用yfinance获取股票数据
    :param stock_code: 股票代码，如 '000001.SZ','BABA'
    :param start_date: 开始日期，格式为 'YYYY-MM-DD'
    :param end_date: 结束日期，格式为 'YYYY-MM-DD'
    :return: 指定时间段内的股票交易数据(DataFrame)
    """
    # 如果soock_code是以.SZ或.SH结尾，则分别替换成.ss或.sz
    if stock_code.endswith('.SZ'):
        stock_code = stock_code.replace('.SZ','.sz')
    elif stock_code.endswith('.SH'):
        stock_code = stock_code.replace('.SH','.ss')
    if start_date == "1990-01-01" and end_date == datetime.now().strftime('%Y-%m-%d'):
        stock_data = yf.download(stock_code, period='max', auto_adjust=True)
    else:
        stock_data = yf.download(stock_code, start=start_date, end=end_date, auto_adjust=True)
    stock_data.columns = ['Close','High','Low','Open','Volume']  #修改列名，YF默认得到的是一个复杂格式的列名，这里统一改为Close,High,Low,Open,Volume，索引是Date(日期)
    return stock_data.dropna()

# 设置股票数据的扩展列
def set_stock_excol(stock_data,col_name):
    '''设置股票数据的扩展列'''
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Log_Returns'] = np.log(1+stock_data['Returns'])
    stock_data['Close_Open'] = stock_data['Close'] - stock_data['Open']
    stock_data['Open_Close'] = stock_data['Open'] - stock_data['Close']
    stock_data['High_Low'] = stock_data['High'] - stock_data['Low']
    stock_data['Diff'] = stock_data['Close'].diff()
    stock_data['Signal'] = np.where(stock_data['Diff'] > 0, 1, 0)
    stock_data['42d']= stock_data['Close'].rolling(42).mean()
    stock_data['252d'] = stock_data['Close'].rolling(252).mean()    
    stock_data['Mov_Vol'] = (stock_data['Log_Returns'].rolling(252).std())*math.sqrt(252)
    return stock_data

def daily_return_ratio(price_list):
    '''每日收益率'''
    # 公式 每日收益率 = (price_t - price_t-1) / price_t-1
    price_list=price_list.to_numpy()
    # 计算每日收益率，从第二个元素开始计算,第一个元素设为NaN
    return np.append(np.nan,(price_list[1:]-price_list[:-1])/price_list[:-1])
    # return (price_list[1:]-price_list[:-1])/price_list[:-1]
    
def daily_return_ratio_log(price_list):
    '''每日对数收益率'''
    # 公式 每日对数收益率 = ln(price_t/price_t-1)
    price_list=price_list.to_numpy()
    # 计算每日对数收益率，从第二个元素开始计算,第一个元素设为NaN
    return np.append(np.nan,np.log(price_list[1:]/price_list[:-1]))
    # return np.log(price_list[1:]/price_list[:-1])
    
# 常用金融资产定价指标
def sum_return_ratio(price_list):
    '''实际总收益率'''
    # 公式 实际总收益率 = (price_t - price_t0) / price_t0
    price_list=price_list.to_numpy()
    return (price_list[-1]-price_list[0])/price_list[0]
def max_draw_down(price_list):
    '''最大回撤率'''
    # 公式 最大回撤率 = (price_t - price_tmax) / price_tmax
    price_list=price_list.to_numpy()
    i = np.argmax((np.maximum.accumulate(price_list) - price_list) / np.maximum.accumulate(price_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(price_list[:i])  # 开始位置
    return (price_list[j] - price_list[i]) / (price_list[j])
def sharpe_ratio(price_list,rf=0.000041):
    '''夏普比率'''
    # 公式 夏普率 = (回报率均值 - 无风险率) / 回报率的标准差
    # pct_change()是pandas里面的自带的计算每日增长率的函数
    daily_return = price_list.pct_change()
    return daily_return.mean()-rf/ daily_return.std()
def information_ratio(price_list,rf=0.000041):
    '''信息比率'''
    # 公式 信息比率 = (总回报率 - 无风险率) / 回报率的标准差
    chaoer=sum_return_ratio(price_list)-((1+rf)**365-1)
    return chaoer/np.std(price_list.pct_change()-rf)
def treynor_ratio(price_list,beta,rf=0.000041):
    '''特雷诺比率'''
    # 公式 特雷诺比率 = (回报率均值 - 无风险率) /  beta
    daily_return = price_list.pct_change()
    return (daily_return.mean()-rf)/beta