import numpy as np
import pandas as pd
import math

class PriceList(object):
    """ 股票数据(Close price)及基本处理 """

    def __init__(self, price_list: pd.Series = None):
        """
        :param price_list: 股票价格序列
        """
        super(PriceList, self).__init__()
        self.sprice_list: pd.Series = pd.Series()
        self.Daily_Return_Ratio: pd.Series = pd.Series()
        self.Daily_Return_Ratio_Log: pd.Series = pd.Series()

        self.Sum_Return_Ratio: np.float64 = None
        self.Max_Draw_Down: np.float64 = None
        self.Sharpe_Ratio: np.float64 = None
        self.Information_Ratio: np.float64 = None
        self.Treynor_Ratio: np.float64 = None
        
        self.Stock_Returns: pd.DataFrame = pd.DataFrame()
        self.Stock_Ratio: dict = {}
        
        if price_list is not None:
            self.bind(price_list)
        
    def sample(self):
        """
        价格列表数据分析示例，默认以阿里巴巴的收盘价格序列为例计算相关参数
        """
        from importlib.resources import files
        from importlib.resources.abc import Traversable
        try:
            # 定位样例BABA的数据文件（适配安装后的路径）
            data_dir: Traversable = files("okfinance") / "data"
            csv_path: Traversable = data_dir / "stock_BABA.csv"
            print(csv_path)
            price_list = pd.read_csv(csv_path)['Close']
            self.Daily_Return_Ratio = self.daily_return_ratio(price_list)
            self.Daily_Return_Ratio_Log = self.daily_return_ratio_log(price_list)
            self.Sum_Return_Ratio = self.sum_return_ratio(price_list)
            self.Max_Draw_Down = self.max_draw_down(price_list)
            self.Sharpe_Ratio = self.sharpe_ratio(price_list)
            self.Information_Ratio = self.information_ratio(price_list)
            self.Treynor_Ratio = self.treynor_ratio(price_list)
                
            print('收益率: daily_return_ratio():',self.Daily_Return_Ratio)
            print('对数收益率: daily_return_ratio_log():',self.Daily_Return_Ratio_Log)
            print('实际总收益率: sum_return_ratio():',self.Sum_Return_Ratio)
            print('最大回撤率: max_draw_down():',self.Max_Draw_Down)
            print('夏普比率: sharpe_ratio(rf=0.000041):',self.Sharpe_Ratio) 
            print('信息比率: information_ratio(rf=0.000041):',self.Information_Ratio)
            print('特雷诺比率: treynor_ratio(beta=1,rf=0.000041):',self.Treynor_Ratio) 
            
            self.Stock_Returns = pd.DataFrame({
                'Close': price_list,
                'Returns': self.Daily_Return_Ratio,             
                'Log_Returns': self.Daily_Return_Ratio_Log

            })
            self.Stock_Ratio = dict(
                Sum_Return_Ratio=self.Sum_Return_Ratio,
                Max_Draw_Down=self.Max_Draw_Down,
                Sharpe_Ratio=self.Sharpe_Ratio,
                Information_Ratio=self.Information_Ratio,
                Treynor_Ratio=self.Treynor_Ratio
            )
            return True
        except Exception as e:
            print(e)
            return False

    def bind(self, price_list: pd.Series, *args, **kwargs):
        """
        :param price_list: 股票价格序列
        """
        try:
            self.price_list = price_list
            self.Daily_Return_Ratio = self.daily_return_ratio(price_list)
            self.Daily_Return_Ratio_Log = self.daily_return_ratio_log(price_list)

            self.Sum_Return_Ratio = self.sum_return_ratio(price_list)
            self.Max_Draw_Down = self.max_draw_down(price_list)
            self.Sharpe_Ratio = self.sharpe_ratio(price_list)
            self.Information_Ratio = self.information_ratio(price_list)
            self.Treynor_Ratio = self.treynor_ratio(price_list)
            
            self.Stock_Returns = pd.DataFrame({
                'Close': price_list,
                'Returns': self.daily_return_ratio,
                'Log_Returns': self.daily_return_ratio_log
            })
            self.Stock_Ratio = dict(
                Sum_Return_Ratio=self.Sum_Return_Ratio,
                Max_Draw_Down=self.Max_Draw_Down,
                Sharpe_Ratio=self.Sharpe_Ratio,
                Information_Ratio=self.Information_Ratio,
                Treynor_Ratio=self.Treynor_Ratio
            )
            return True
        except Exception as e:
            print(e)
            return False
        
    @classmethod
    def daily_return_ratio(self, price_list):
        '''每日收益率'''
        # 公式 每日收益率 = (price_t - price_t-1) / price_t-1
        price_list=price_list.to_numpy()
        # 计算每日收益率，从第二个元素开始计算,第一个元素设为NaN
        return np.append(np.nan,(price_list[1:]-price_list[:-1])/price_list[:-1])
        # return (price_list[1:]-price_list[:-1])/price_list[:-1]
        
    @classmethod
    def daily_return_ratio_log(self, price_list):
        '''每日对数收益率'''
        # 公式 每日对数收益率 = ln(price_t/price_t-1)
        price_list=price_list.to_numpy()
        # 计算每日对数收益率，从第二个元素开始计算,第一个元素设为NaN
        return np.append(np.nan,np.log(price_list[1:]/price_list[:-1]))
        # return np.log(price_list[1:]/price_list[:-1])
        
    # 常用金融资产定价指标
    @classmethod
    def sum_return_ratio(self, price_list):
        '''实际总收益率'''
        # 公式 实际总收益率 = (price_t - price_t0) / price_t0
        price_list=price_list.to_numpy()
        return (price_list[-1]-price_list[0])/price_list[0]
    
    @classmethod
    def max_draw_down(self, price_list):
        '''最大回撤率'''
        # 公式 最大回撤率 = (price_t - price_tmax) / price_tmax
        price_list=price_list.to_numpy()
        i = np.argmax((np.maximum.accumulate(price_list) - price_list) / np.maximum.accumulate(price_list))  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(price_list[:i])  # 开始位置
        return (price_list[j] - price_list[i]) / (price_list[j])
    
    @classmethod
    def sharpe_ratio(self, price_list, rf=0.000041):
        '''夏普比率'''
        # 公式 夏普率 = (回报率均值 - 无风险率) / 回报率的标准差
        # pct_change()是pandas里面的自带的计算每日增长率的函数
        daily_return = price_list.pct_change()
        return daily_return.mean()-rf/ daily_return.std()*math.sqrt(252)
    
    @classmethod
    def information_ratio(self, price_list, rf=0.000041):
        '''信息比率'''
        # 公式 信息比率 = (总回报率 - 无风险率) / 回报率的标准差
        chaoer=self.sum_return_ratio(price_list)-((1+rf)**365-1)
        return chaoer/np.std(price_list.pct_change()-rf)*math.sqrt(252)
    
    @classmethod
    def treynor_ratio(self, price_list, beta=1, rf=0.000041):
        '''特雷诺比率'''
        # 公式 特雷诺比率 = (回报率均值 - 无风险率) /  beta
        daily_return = price_list.pct_change()
        return (daily_return.mean()-rf)/beta