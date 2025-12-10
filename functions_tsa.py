
#%%
### 完整的单只股票历史数据分析过程
import math, datetime, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts # 导入tushare
import scipy.stats as scs
import statsmodels.api as sm

## 获取相关全局参数
# 初始化pro接口(请个人去tushare官网免费申请自用的token并替换)
pro = ts.pro_api('our_tushare_token')

def tsa(stock_code='000651.SZ', start_date='', end_date=''):
    if stock_code in [""," ",None]:
        tscode = '000651.SZ'
        print('未输入股票代码, 默认使用格力电器（代码为: 000651.SZ）')
    else:
        tscode = stock_code
        print('您输入的股票代码为: {}'.format(stock_code))
    if start_date in [""," ",None]:
        start_date = '19900101' # 起始时间, 中国股市起始时间是1990年12月1日
        print('未输入起始时间, 默认使用19900101')
    if end_date in [""," ",None]:
        end_date = datetime.datetime.now().strftime('%Y%m%d')
        print('未输入结束时间, 默认使用当前时间: {}'.format(end_date))
    start = start_date.replace('-','')
    end = end_date.replace('-','')

    # 参考代码 (股票代码，股票名称)(为AI自动生成, 不一定正确, 请注意甄别)
    stock_kv = {'600755.SH': '厦门国贸', '600309.SH': '万华化学', '600031.SH': '三一重工', '600886.SH': '国投电力', '000768.SZ': '中航飞机', '002594.SZ': '比亚迪', '000651.SZ': '格力电器', '600138.SH': '中青旅', '600177.SH': '雅戈尔', '600276.SH': '恒瑞医药', '600519.SH': '贵州茅台', '002714.SZ': '牧原股份', '600036.SH': '招商银行', '601318.SH': '中国平安', '000002.SZ': '万科A', '601138.SH': '工业富联', '600111.SH': '北方稀土', '300413.SZ': '芒果超媒', '002093.SZ': '国脉科技', '600900.SH': '长江电力'}
    # 如果没有输入股票代码，则默认使用格力电器（代码为: 000651.SZ）
    
    tsname = stock_kv.get(tscode, '未知')
    if tsname != '未知':
        print('系统识别到股票代码为 {} 的股票名称为: {}'.format(tscode,tsname))
        
    stock_file = './data/stock_{}_{}_{}.csv'.format(tscode,start,end)
    stock_result_file = './data/stock_{}_{}_{}_result.csv'.format(tscode,start,end)
    # 如果文件夹下不存在data子文件夹则手动创建或换成下面的路径
    #stock_file = 'stock_{}_{}_{}.csv'.format(tscode,start,end)
    #stock_result_file = 'stock_{}_{}_{}_result.csv'.format(tscode,start,end)

    ## 通过tushare接口获取股票数据并保存到csv文件

    if not os.path.exists(stock_file):
        print('正在获取股票代码为 {} {} 的\n从 {} 到 {} 的日线数据...'.format(tscode,tsname,start,end))
        # 获取指定股票指定日期的日线数据, 注意股指代码可能需要额外权限
        data = pro.daily(ts_code=tscode, start_date=start, end_date=end)
        # print(data.describe())
        # print(data.head())
        # print(data.tail())
        print('\n获取股票代码为 {} {} 的\n从 {} 到 {} 的\n有效交易数据和共 {} 条,\n已存入 {} 文件中.'.format(tscode,tsname,start,end,len(data),stock_file))
        data.to_csv(stock_file,columns=['trade_date','open','high','low','close','vol']) #选择保存
    else:
        #从csv读取数据
        data = pd.read_csv(stock_file, encoding = "gbk")
        print('股票代码为 {} {} 的\n从 {} 到 {} 的\n日线数据共 {} 条\n已经存在于{}文件中,\n正在读入到内存中...'.format(tscode,tsname,start,end,len(data),stock_file))

    ## 读取数据并进行预处理
    # 如果是使用tushare获取的股票数据，需要将日期顺序颠倒并将close,open,trade_date列更名为Close,Open,Date
    data.dropna(inplace=True) #去掉缺失值, inplace=True表示在原数据上修改
    data1=pd.DataFrame()
    data1['Date']=pd.to_datetime(data['trade_date'],format='%Y%m%d')
    data1['Close']=data['close']
    data1['Open']=data['open']
    # 将数据框data1顺序颠倒
    data1 = data1.iloc[::-1]
    data1.index = data1['Date']
    del data1['Date']
    print('股票代码为 {} {} 的\n从 {} 到 {} 的日线数据已经预处理完成, \n其最后三行示例如下:'.format(tscode,tsname,start,end))
    # print(data1.describe())
    # print(data1.head())
    print(data1.tail(3))

    ## 时间序列分析（股票数据，对数收益率, 移动历史波动率, 42D与252D移动平均等数据）

    # 计算42d, 252d 移动平均
    data1['42d']= data1['Close'].rolling(42).mean()
    data1['252d'] = data1['Close'].rolling(252).mean()
    print('\n股票代码为 {} {} 的股票的42D与252D移动平均数据:\n'.format(tscode,tsname))
    print(data1[['Close','42d','252d']].tail())
    data1[['Close','42d','252d']].plot(figsize=(8,5))
    plt.show()
    # 计算收盘价的对数收益率(Log_Rets), 以及收盘价与开盘价的差值(CloseOpen); shift(1)是向下移动一行, 即当前行数据减去上一行数据
    data1['Log_Rets']=np.log(data1['Close']/data1['Close'].shift(1))
    data1['CloseOpen'] = data1['Close']-data1['Open']
    # 计算移动历史波动率(Mov_Vol: moving annual volatility) 
    data1['Mov_Vol'] = (data1['Log_Rets'].rolling(252).std())*math.sqrt(252)
    print('\n股票代码为 {} {} 的股票的对数收益率, 移动历史波动率等数据:\n'.format(tscode,tsname))
    print(data1[['Close','CloseOpen','Log_Rets','Mov_Vol']].tail())
    data1[['Close','Mov_Vol','CloseOpen','Log_Rets']].plot(subplots=True, style='b',figsize=(8,5))
    plt.show()
    # 将计算结果保存到csv
    print('\n股票代码为 {} {} 的时间序列分析相关结果已保存到\n{}文件中.'.format(tscode,tsname,stock_result_file))
    print(data1.describe())
    data1.to_csv(stock_result_file) #保存下载的数据

    ## 正态分布检测（直方图与QQ图等）

    #将对数收益率转换为数组
    log_array = np.array(data1['Log_Rets'].dropna())
    # 绘制直方图
    print('\n股票代码为 {} {} 的对数收益率的直方图:'.format(tscode,tsname))
    data1['Log_Rets'].dropna().hist(bins=50)
    plt.show()
    # 输出统计量
    print('\n股票代码为 {} {} 的对数收益率的统计量:\n'.format(tscode,tsname))
    sta = scs.describe(log_array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * "-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))
    # 绘制QQ图
    print('\n股票代码为 {} {} 的对数收益率的QQ图:'.format(tscode,tsname))
    sm.qqplot(data1['Log_Rets'].dropna(), line='s')
    plt.show()
    # 输出偏度、峰度、正态性检验
    print('\n股票代码为 {} {} 的对数收益率的偏度、峰度、正态性检验:\n'.format(tscode,tsname))
    print("Skew of data set  %14.3f" % scs.skew(log_array))
    print("Skew test p-value %14.3f" % scs.skewtest(log_array)[1])
    print("Kurt of data set  %14.3f" % scs.kurtosis(log_array))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(log_array)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(log_array)[1])

    print('\n股票代码为 {} {} 的历史数据分析全部完成.'.format(tscode,tsname))
    return stock_result_file
