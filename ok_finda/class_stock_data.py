import numpy as np
import pandas as pd
import yfinance as yf
import tushare as ts
import math
from datetime import datetime

class StockData(object):
    """ 股票数据及基本处理 """
    from importlib.resources import files
    from importlib.resources.abc import Traversable
    # 定位样例BABA的数据文件（适配安装后的路径）
    data_dir: Traversable = files("ok_finda") / "data"
    csv_path: Traversable = data_dir / "stock_BABA.csv"
    stock_data_baba = pd.read_csv(csv_path).set_index('Date')

    def __init__(self, source: str = "yfinance", *args, **kwargs):
        """
        :param source: 数据来源 ，默认yfinance，可选tushare
        """
        self.source = source
        self.token = None  # 初始化默认是空的

    @classmethod
    # 获取csv格式股票数据（没有数据文件的话自动下载）
    def load_stock_csv(self, stock_code, start_date=None, end_date=None, csv_file=None):
        """
        加载股票数据，先尝试从文件加载，若文件不存在则从指定源下载并保存。
        :param stock_code: 股票代码，如 '601318.SH'
        :param start_date: 开始日期，格式为 'YYYY-MM-DD'
        :param end_date: 结束日期，格式为 'YYYY-MM-DD'
        :param output_file: 输出的pickle文件路径,会自动加上./data/前缀
        :param source: 数据来源，'yf'表示yfinance，'ts'表示tushare，默认'yf'
        :return: 指定时间段内的股票交易数据(DataFrame)
        """
        source = self.source
        token = self.token
        
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
                df = self.get_stock_data_ts(stock_code, start_date, end_date, token=token)
            elif source == 'yfinance':
                df = self.get_stock_data_yf(stock_code, start_date, end_date)
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
    @classmethod
    def get_stock_data_ts(self, stock_code, start_date, end_date, token=None):
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
    @classmethod
    def get_stock_data_yf(self, stock_code, start_date, end_date):
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
    @classmethod
    def set_stock_excol(self, stock_data,col_name):
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
    