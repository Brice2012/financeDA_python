# ##################################################
# okfinance package 2.0
# Financial Data Analysis - Portfolio Optimization
# 金融数据分析 - 投资组合优化
# Author: YeJunjie (Brice)
# E_Mail: ye@okwords.cn
# Date: 2025-12-20
# ####################################################

# 投资组合优化

# 导入相关的包
# pip install numpy pandas yfinance matplotlib scipy statsmodels # 安装需要的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import scipy.optimize as sco
import statsmodels.api as sm
from .class_stock_data import StockData
from .functions_stat import stat_describe, stat_norm_tests, stat_gen_paths

class StockPO(object):
    """ 投资组合优化（多支股票） """
    
    def __init__(self, stocks: dict = None, source: str = "yfinance", token: str = None,  *args, **kwargs):
        """
        :param stocks: 股票数据字典，键为股票代码，值为股票名称
        """
        super(StockPO, self).__init__()
        self.source = source
        self.token = token
        self.stocks = stocks
        self.symbols = []
        self.names = []
        self.noa:int = 0
        self.data = pd.DataFrame()
        self.rets = pd.DataFrame()
        self.mean_rets = None  #年化收益率。每年252个交易日
        self.cov_rets = None   #协方差矩阵，是投资组合选择过程的核心部分。
        # 加载股票数据
        if stocks is not None:
            self.symbols = list(stocks.keys())
            self.names = list(stocks.values())
            self.noa = len(self.symbols)
            self.load_podata(self.symbols, source=source, token=token)
        else:
            self.load_sample()
        
        # 使用mc模拟计算
        self.prets = np.array([]) #预期收益率
        self.pvols = np.array([]) #预期波动率
        self.po_mc()
        
        # 计算ef和cml时需要的参数
        self.cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  # 约束条件：权重总和为1
        self.bnds = tuple((0, 1) for x in range(self.noa))  # 权重的取值范围
        self.weights0 = self.noa * [1. / self.noa,]  # 初始权重
        self.bnds0 = tuple((0, 1) for x in self.weights0)  # 权重的取值范围
        
        self.opts = None # 最大夏普指数情况下的最优权重
        self.optv = None # 最小波动率情况下的最优权重
        self.opts, self.opts_r = self.po_max_sharpe()
        self.optv, self.optv_r = self.po_min_vol()
        
        # 有效边界（efficient frontier）
        self.trets = np.linspace(0.06, 0.15, 30) # 默认的目标收益率水平 (0, 0.25, 50)
        #self.trets = None # 目标收益率水平        
        self.tvols = None # 目标收益率水平下的最小波动率（po_ef)
        self.po_ef(trets=self.trets)
        self.po_cml()

    def set_token(self, token: str) -> bool:
        """
        设置tushare token
        :param token: tushare token
        """
        if self.source != 'tushare':
            print(f'数据源 {self.source} 不是tushare，w无需设置token')
            return False
        else:
            self.token = token
            return True

    def load_sample(self):
        """ 加载默认样例数据 """
        #设置默认样例数据
        from importlib.resources import files
        from importlib.resources.abc import Traversable
        self.stocks = {'600031.SH': '三一重工', '601138.SH': '工业富联', '000768.SZ': '中航西飞', '600519.SH': '贵州茅台'}
        self.symbols = list(self.stocks.keys())
        self.names = list(self.stocks.values())
        try:
            data_dir: Traversable = files("okfinance") / "data"
            csv_path: Traversable = data_dir / "stock_close_2025.csv"
            data = pd.read_csv(csv_path, encoding="gbk", index_col=0, parse_dates=True)[self.symbols]  # 从csv读取数据
            self.data = data.dropna()
            self.noa = len(self.symbols)
            self.rets = np.log(self.data / self.data.shift(1))
            self.mean_rets = self.rets.mean() * 252
            self.cov_rets = self.rets.cov() * 252
            print(f'\n\n默认样例数据 {self.stocks}加载成功， \n包含 {self.noa} 支股票')
            print(f'前三行数据: \n{self.data.head(3)}')
            return True
        except FileNotFoundError:
            print("文件未找到，请检查文件路径是否正确。")
            return False
        except Exception as e:
            print(f"发生未知错误: {e}")
            return False

    def load_podata(self, symbols: list = None, source: str = "yfinance", token: str = None, save: bool = True):
        """ 加载股票数据 """
        try:
            data =pd.DataFrame()
            for symbol in symbols:
                data[symbol] = StockData(stock_code=symbol, source=source, token=token).DF['Close']
            if save:
                data.to_csv(f"stock_po.csv", index=True)
            self.data = data.dropna()
            self.noa = len(self.symbols)
            self.rets = np.log(self.data / self.data.shift(1))
            self.mean_rets = self.rets.mean() * 252
            self.cov_rets = self.rets.cov() * 252
            print(f'\n\n{self.stocks}加载成功， \n包含 {self.noa} 支股票')
            print(f'前三行数据: \n{self.data.head(3)}')
            return True
        except Exception as e:
            print(f"发生错误: {e}")
            return False
        
    # 计算一组随机权重的预期收益和预期波动率
    def po_random(self):
        """
        计算一组随机权重的预期收益和预期波动率
        """
        print('\n\n年化收益率: ')
        print(self.mean_rets)  #年化收益率。每年252个交易日
        print('\n协方差矩阵: ')
        print(self.cov_rets)  #协方差矩阵，是投资组合选择过程的核心部分。

        print('\n\n某组随机权重的预期收益和预期波动率: ')
        weights = np.random.random(self.noa)
        weights /= np.sum(weights)  #保证权重的总和为1
        print(f'\n权重: {weights}')
        pret = np.sum(self.rets.mean() * weights) * 252
        print(f'\n预期收益: {pret:.4f}')  #预期收益
        pvar = np.dot(weights.T, np.dot(self.cov_rets, weights))  #预期投资组合方差
        print(f'\n预期投资组合方差: {pvar:.4f}')
        pvol = np.sqrt(np.dot(weights.T, np.dot(self.cov_rets, weights))) #预期波动率（投资组合标准差）
        print(f'\n预期波动率(投资组合标准差): {pvol:.4f}')  #预期波动率（投资组合标准差）
        psharper = pret / pvol
        print(f'\n夏普比率: {psharper:.4f}')  #夏普比率
        # 将pret, pvar, pvol, psharpe组合成一个数组，以便后续使用
        results = np.array([pret, pvar, pvol, psharper]).T
        return weights, results

    def po_weights(self, weights):
        """
        计算一组确定权重的预期收益和预期波动率
        """
        weights = np.array(weights)
        print(f'\n\n权重为{weights}的预期收益和预期波动率: ')
        weights /= np.sum(weights)  #保证权重的总和为1
        print(f'\n调整归一后的权重: {weights}')
        pret = np.sum(self.rets.mean() * weights) * 252
        print(f'\n预期收益: {pret:.4f}')  #预期收益
        pvar = np.dot(weights.T, np.dot(self.cov_rets, weights))  #预期投资组合方差
        print(f'\n预期投资组合方差: {pvar:.4f}')
        pvol = np.sqrt(np.dot(weights.T, np.dot(self.cov_rets, weights))) #预期波动率（投资组合标准差）
        print(f'\n预期波动率(投资组合标准差): {pvol:.4f}')  #预期波动率（投资组合标准差）
        psharper = pret / pvol
        print(f'\n夏普比率: {psharper:.4f}')  #夏普比率
        # 将pret, pvar, pvol, psharpe组合成一个数组，以便后续使用
        results = np.array([pret, pvar, pvol, psharper]).T
        return weights, results

    def po_mc(self, num=3700):
        """
        随机模拟num_simulations次montecarlo的预期收益和预期波动率
        """
        ###蒙特卡洛模拟。预期的投资组合收益和方差
        prets = [] # 预期收益
        pvols = [] # 预期波动率
        for p in range (num):
            weights = np.random.random(self.noa) # 随机权重
            weights /= np.sum(weights)      # 权重总和为1
            prets.append(np.sum(self.rets.mean() * weights) * 252)  # 预期收益
            pvols.append(np.sqrt(np.dot(weights.T, 
                                        np.dot(self.cov_rets, weights)
                                        )))  # 预期波动率
        prets = np.array(prets)
        pvols = np.array(pvols)
        self.prets = prets
        self.pvols = pvols

        # 下图显示了2500个随机投资组合的预期收益和预期波动率
        plt.figure(figsize=(8, 4))
        plt.scatter(pvols, prets, c=prets / pvols, marker='o')
        plt.grid(True)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()
        return prets, pvols

    # 计算最小波动率和最大夏普指数条件下的最优投资组合
    #定义函数，根据权重计算收益和波动率
    def statistics(self, weights):
        weights = np.array(weights) # 权重
        pret = np.sum(self.rets.mean() * weights) * 252  # 预期收益
        pvol = np.sqrt(np.dot(weights.T, np.dot(self.cov_rets, weights)))  # 预期波动率
        return np.array([pret, pvol, pret / pvol])

    # opts 为最大夏普指数情况下的最优投资组合
    def po_max_sharpe(self):
        
        # 最大化夏普指数（最小化夏普指数的负值）
        def min_func_sharpe(weights):
            return -self.statistics(weights)[2]
        
        opts = sco.minimize(min_func_sharpe, self.noa * [1. / self.noa,], method='SLSQP', bounds=self.bnds, constraints=self.cons) # SLSQP是一种优化算法,Sequential Least SQuares Programming, 顺序最小二乘法
        print('\n\n最大夏普指数情况下: ')
        print(opts)
        print('\n最优投资组合为: ')
        print(self.symbols)
        print(opts['x'].round(3))
        print('\n最优投资组合情况下的预期收益率、波动率和最优夏普指数分别为: ')
        results = self.statistics(opts['x']).round(3)
        print(results)
        self.opts = opts
        self.opts_r = results
        return opts,results

    # optv 为最小波动率情况下的最优投资组合
    def po_min_vol(self):
        
        # 最小化投资组合的方差，即最小化波动率
        def min_func_variance(weights):
            return self.statistics(weights)[1] ** 2
        
        optv = sco.minimize(min_func_variance, self.noa * [1. / self.noa,], method='SLSQP', bounds=self.bnds, constraints=self.cons) 
        print('\n\n最小波动率情况下: ')
        print(optv)
        print('\n最优投资组合为: ')
        print(optv['x'].round(3))
        print('\n最优投资组合情况下的预期收益率、波动率和最优夏普指数分别为: ')
        results = self.statistics(optv['x']).round(3)
        print(results)
        self.optv = optv
        self.optv_r = results
        return optv,results


    # 有效边界（Efficient Frontier）
    # 给定目标收益率水平下的最小波动率投资组合 min_vol for given tret
    def po_ef(self, trets=None):
        
        def min_func_port(weights):
            return self.statistics(weights)[1]
        
        # constraints: 两个约束条件，一个是给定目标收益率，另一个是权重总和为1
        # 注意，如果后面画资本市场线时报数据错误，下面这三个参数需要修改，
        # trets是一个列表，默认是(0, 0.25, 50), 可以观察前后有没有垂直出现的x来调整取值。
        if trets is None:
            trets = np.linspace(0, 0.25, 50) # 目标收益率水平
        tvols = [] # 目标收益率水平下的最小波动率
        for tret in trets:
            cons = ({'type': 'eq', 'fun': lambda x:  self.statistics(x)[0] - tret},
                    {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}) # 约束条件
            res = sco.minimize(min_func_port, self.noa * [1. / self.noa,], method='SLSQP', bounds=self.bnds0 , constraints=cons) # 最小化波动率
            tvols.append(res['fun']) # 最小波动率
        tvols = np.array(tvols) # 目标收益率水平下的最小波动率
        opts,r = self.po_max_sharpe() # 最大夏普指数情况下的最优投资组合
        optv,r = self.po_min_vol()    # 最小波动率情况下的最优投资组合
        print('\n有效边界，左侧*表示给定收益率水平的最小方差/波动率的投资组合，另一个*表示最大夏普指数的投资组合。')
        plt.figure(figsize=(8, 4))
        plt.scatter(self.pvols, self.prets,
                    c=self.prets / self.pvols, marker='o')
                    # random portfolio composition
        plt.scatter(tvols, trets,
                    c=trets / tvols, marker='x')
                    # efficient frontier
        plt.plot(self.statistics(opts['x'])[1], self.statistics(opts['x'])[0],
                'r*', markersize=15.0)
                    # portfolio with highest Sharpe ratio
        plt.plot(self.statistics(optv['x'])[1], self.statistics(optv['x'])[0],
                'y*', markersize=15.0)
                    # minimum variance portfolio
        plt.grid(True)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()
        self.trets = trets
        self.tvols = tvols
        return trets, tvols

    # 资本市场线（Capital Market Line）
    def po_cml(self):
        import scipy.interpolate as sci
        trets, tvols = self.po_ef(self.trets)
        ind = np.argmin(tvols) # 最小波动率的索引
        evols = tvols[ind:] # 有效边界的波动率
        erets = trets[ind:] # 有效边界的收益率

        tck = sci.splrep(evols, erets) # 样条插值,返回一个三元组(t,c,k),其中t是节点，c是系数，k是阶数

        def f(x):
            ''' Efficient frontier function (splines approximation). 有效边界函数（样条逼近） '''
            return sci.splev(x, tck, der=0) # 样条插值；der=0表示零阶导数
        def df(x):
            ''' First derivative of efficient frontier function. 有效边界函数的一阶导数 '''
            return sci.splev(x, tck, der=1) # 样条插值；der=1表示一阶导数

        def equations(p, rf=0.02):
            eq1 = rf - p[0] # 无风险利率
            eq2 = rf + p[1] * p[2] - f(p[2]) # 资本市场线
            eq3 = p[1] - df(p[2]) # 资本市场线的斜率
            return eq1, eq2, eq3

        opt = sco.fsolve(equations, [0.01, 0.6, 0.10]) # 无风险利率、资本市场线的斜率和截距，比如rf=0.01, p=(0.01, 0.5, 0.15)
        #上述参数（包括equations函数）需要反复尝试，组合各种合理的猜测。

        print('\n资本市场线。\n数值优化结果')
        print(opt)
        print('\n方程是否合乎预期（值是否为零）')
        print(np.round(equations(opt), 6))
        print('\n下图星号代表有效边界中切线穿过无风险资产点。即无风险利率为1%时的资本市场线和相切的投资组合。')
        plt.figure(figsize=(8, 4))
        plt.scatter(self.pvols, self.prets,
                    c=(self.prets - 0.01) / self.pvols, marker='o')
                    # random portfolio composition
        plt.plot(evols, erets, 'g', lw=4.0)
                    # efficient frontier
        cx = np.linspace(0.0, 0.3) # 有效边界的波动率范围
        plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
                    # capital market line
        plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0) 
        plt.grid(True)
        plt.axhline(0, color='k', ls='--', lw=2.0)
        plt.axvline(0, color='k', ls='--', lw=2.0)
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

        # constraints: 两个约束条件，一个是给定目标收益率，另一个是权重总和为1
        cons = ({'type': 'eq', 'fun': lambda x:  self.statistics(x)[0] - f(opt[2])},
                {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}) # 约束条件
        # sco.minimize函数
        # min_func_port是目标函数，即波动率, noa * [1. / noa,]是初始权重, method='SLSQP'是优化算法, bounds=bnds是权重的取值范围, constraints=cons是约束条件
        
        def min_func_port(weights):
            return self.statistics(weights)[1]
        
        res = sco.minimize(min_func_port, self.noa * [1. / self.noa,], method='SLSQP',bounds=self.bnds0, constraints=cons)
        print('\n最优投资组合的权重')
        print(res['x'].round(3))
        print('\n最优投资组合情况下的预期收益率、波动率和最优夏普指数分别为：')
        print(self.statistics(res['x']).round(3))
