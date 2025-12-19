# ##################################################
# financeda package 2.0
# Financial Data Analysis
# 金融数据分析
# Author: YeJunjie (Brice)
# E_Mail: ye@okwords.cn
# Date: 2025-12-20
# ####################################################

# 导入需要的包
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr

S0 = 100.0        # initial index level; 初始指数水平
r = 0.05         # constant short rate; 短期利率常数
sigma = 0.25     # constant volatility; 波动率常数
T = 1.0          # in years; 期权到期时间
I = 50000        # number of simulations; 模拟次数
M = 50           # number of time steps; 时间步数

# gen_sn函数用于生成模拟的随机数
def gen_sn(M, I, anti_paths=True, mo_match=True):
    ''' Function to generate random numbers for simulation. 
    Parameters
    ==========
    M : int; number of time intervals for discretization # 时间间隔数
    I : int; number of paths to be simulated # 模拟路径数
    anti_paths: boolean; use of antithetic variates # 是否使用反向路径
    mo_math : boolean; use of moment matching # 是否使用矩匹配
    '''
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2))) # half paths; 半路径
        sn = np.concatenate((sn, -sn), axis=1)  # antithetic variates; 反向路径
    else:
        sn = npr.standard_normal((M + 1, I)) # full paths; 全路径
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std() # moment matching; 矩匹配
    return sn


# 欧式期权 European Options
# 风险中立预期定价
# gbm_mcs_stat函数用于估算欧式期权的风险中立预期定价, 使用蒙特卡洛模拟(到期时的指数水平)
def gbm_mcs_stat(K):
    ''' Valuation of European call option in Black-Scholes-Merton
    by Monte Carlo simulation (of index level at maturity)
    Parameters
    ==========
    K : float; (positive) strike price of the option  # 行权价
    Returns
    =======
    C0 : float; estimated present value of European call option # 欧式看涨期权的风险中立预期定价
    '''
    sn = gen_sn(1, I) # generate random numbers; 生成随机数
    # simulate index level at maturity; 模拟到期时的指数水平
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T 
                 + sigma * np.sqrt(T) * sn[1])
    # calculate payoff at maturity; 计算到期时的回报
    hT = np.maximum(ST - K, 0)
    # calculate MCS estimator; 计算蒙特卡洛估算值
    C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
    return C0

# a0=gbm_mcs_stat(K=105.)
# print("考虑行权价K=105时的风险中立预期定价为(%.4f)"%(a0))

#%%
# 看涨和看跌期权的价格估算
# gbm_mcs_dyna函数用于估算欧式期权的风险中立预期定价, 使用蒙特卡洛模拟(指数水平路径)
def gbm_mcs_dyna(K, option='call'):
    ''' Valuation of European options in Black-Scholes-Merton
    by Monte Carlo simulation (of index level paths)
    Parameters
    ==========
    K : float; (positive) strike price of the option # 行权价
    option : string; type of the option to be valued ('call', 'put')   # 期权类型(看涨、看跌)
    Returns
    =======
    C0 : float; estimated present value of European call option # 欧式看涨期权的风险中立预期定价
    '''
    dt = T / M # length of time interval; 时间间隔长度
    # simulation of index level paths
    S = np.zeros((M + 1, I)) # index level matrix; 指数水平矩阵
    S[0] = S0 # initial index level; 初始指数水平
    sn = gen_sn(M, I) # random numbers; 随机数
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
                + sigma * np.sqrt(dt) * sn[t])  # index values at time t; 时间t的指数水平
    # case-based calculation of payoff; 基于情况的回报计算
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    # calculation of MCS estimator; 计算蒙特卡洛估算值
    C0 = np.exp(-r * T) * 1 / I * np.sum(hT) 
    return C0

# a1=gbm_mcs_dyna(K=110., option='call')
# a2=gbm_mcs_dyna(K=110., option='put')
# print("相同行权价K=110的看涨(%.4f)和看跌（%.4f)期权的价格估算"%(a1,a2))

# 美式期权 American Options
# gbm_mcs_amer函数用于估算美式期权的风险中立预期定价, 使用蒙特卡洛模拟(指数水平路径)
def gbm_mcs_amer(K, option='call'):
    ''' Valuation of American option in Black-Scholes-Merton
    by Monte Carlo simulation by LSM algorithm
    
    Parameters
    ==========
    K : float
        (positive) strike price of the option # 行权价
    option : string
        type of the option to be valued ('call', 'put') # 期权类型(看涨、看跌)
    
    Returns
    =======
    C0 : float
        estimated present value of European call option # 美式看涨期权的风险中立预期定价
    '''
    dt = T / M # length of time interval; 时间间隔长度
    df = np.exp(-r * dt) # discount factor per time interval; 每个时间间隔的贴现因子
    # simulation of index levels; 指数水平的模拟
    S = np.zeros((M + 1, I)) # index level matrix; 指数水平矩阵
    S[0] = S0 # initial index level; 初始指数水平
    sn = gen_sn(M, I) # random numbers; 随机数
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
                + sigma * np.sqrt(dt) * sn[t])  # index values at time t; 时间t的指数水平
    # case based calculation of payoff; 基于情况的回报计算
    if option == 'call':
        h = np.maximum(S - K, 0)
    else:
        h = np.maximum(K - S, 0)
    # LSM algorithm; LSM算法
    V = np.copy(h) # value matrix; 价值矩阵
    for t in range(M - 1, 0, -1):
        reg = np.polyfit(S[t], V[t + 1] * df, 7) # polynomial regression; 多项式回归
        C = np.polyval(reg, S[t]) # evaluation of polynominal; 多项式评估
        V[t] = np.where(C > h[t], V[t + 1] * df, h[t]) # exercise decision; 行权决策
    # MCS estimator; MCS估算值
    C0 = df * 1 / I * np.sum(V[1]) # LSM estimator; LSM估算值
    return C0

# a1=gbm_mcs_amer(110., option='call')
# a2=gbm_mcs_amer(110., option='put')
# print("美式期权:行权价K=110时看涨(%.4f)和看跌（%.4f)期权的价格估算"%(a1,a2))

# 估算期权溢价（美式期权与欧式期权的差价）
def option_premium(k_list=np.arange(80., 120.1, 5.), option='put'):
    euro_res = []
    amer_res = []
    # k_list = np.arange(80., 120.1, 5.) # strike price list; 行权价列表
    for K in k_list:
        euro_res.append(gbm_mcs_dyna(K, option)) # 欧式看跌期权的风险中立预期定价
        amer_res.append(gbm_mcs_amer(K, option)) # 美式看跌期权的风险中立预期定价
    euro_res = np.array(euro_res)
    amer_res = np.array(amer_res)

    # 画图比较欧式和美式期权的价格
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(k_list, euro_res, 'b', label='European put')
    ax1.plot(k_list, amer_res, 'ro', label='American put')
    ax1.set_ylabel('call option value')
    ax1.grid(True)
    ax1.legend(loc=0)
    wi = 1.0
    ax2.bar(k_list - wi / 2, (amer_res - euro_res) / euro_res * 100, wi)
    ax2.set_xlabel('strike')
    ax2.set_ylabel('early exercise premium in %')
    ax2.set_xlim(left=75, right=125)
    ax2.grid(True)
    # tag: opt_euro_amer
    # title: Comparsion of European and LSM Monte Carlo estimator values
    plt.show()
    return euro_res, amer_res