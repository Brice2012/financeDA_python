#
# Valuation of European call options in Black-Scholes-Merton model
# incl. Vega function and implied volatility estimation
# bsm_functions.py
#

# Analytical Black-Scholes-Merton (BSM) Formula


def bsm_call_value(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model.
    Analytical formula.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    
    Returns
    =======
    value : float
        present value of the European call option
    '''
    from math import log, sqrt, exp
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
            - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
    # stats.norm.cdf --> cumulative distribution function
    #                    for normal distribution
    return value

# Vega function


def bsm_vega(S0, K, T, r, sigma):
    ''' Vega of European option in BSM model.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    
    Returns
    =======
    vega : float
        partial derivative of BSM formula with respect
        to sigma, i.e. Vega
    '''
    from math import log, sqrt
    from scipy import stats

    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega

# Implied volatility function


def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    ''' Implied volatility of European call option in BSM model.
    
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma_est : float
        estimate of impl. volatility
    it : integer
        number of iterations
    
    Returns
    =======
    simga_est : float
        numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
                        / bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est

##### 隐含波动率、BSM模型、期权定价 相关知识#####
"""
# 期权定价

**金融衍生工具**是从另一项基础资产中获取价值的工具，衍生产品类型包括远期，期货，掉期和期权等。**期权**是指一种合约，该合约赋予持有人在某一特定日期或该日期之前的任何时间以固定价格购进或售出一种资产的权利，是衍生工具的一种。由于期权交易方式、方向、标的物的不同，产生了众多的期权品种。
**欧式期权和美式期权**：美式期权的买方可以在到期日或之前任一交易日提出执行合约，而欧式期权只能在到期日行权。因此，美式期权的买方权利相对较大。不过美式期权的卖方风险相应也较大。所以同样条件下，美式期权的价格也相对较高。在美国交易的期权大部分是美式的。但是，外汇期权以及一些股票指数期权显然是个例外。
常见的**期权定价模型**有两种。一是著名的Black Scholes期权定价模型（假定连续的随机过程），二是Cox-Ross-Rubinstein期权定价模型（假定离散的随机过程）。

## Black Scholes期权定价模型

布莱克-舒尔斯模型（Black-Scholes Model），简称BS模型，是一种为期权或权证等金融衍生工具定价的数学模型，由美国经济学家迈伦·舒尔斯（Myron Scholes）与费雪·布莱克（Fischer Black）首先提出，并由罗伯特·墨顿（Robert C. Merton）完善。该模型就是以迈伦·舒尔斯和费雪·布莱克命名的。1997年迈伦·舒尔斯和罗伯特·墨顿凭借该模型获得诺贝尔经济学奖。然而它假设价格的变动，会符合高斯分布（即俗称的钟形曲线），但在财务市场上经常出现符合统计学肥尾现象的事件，这影响此公式的有效性。

Black Scholes期权定价公式作了一些假设。

- 首先是市场没有套利。这意味着不可能有价格差异。
- 第二个假设是基础资产价格遵循布朗运动。
- 第三个假设表明基础股票不支付任何股息。（**因此对于有股息的情况需要做相应处理**）
- 第四个假设是不涉及交易成本，并且可以以任何分数进行基础股票的买卖。
- 最后一个假设是我们知道短期利率，并且该利率随时间是恒定的。

BS 模型对市场的假设有：

- 两种资产：债券（无风险资产）、股票（风险资产）；
- No Arbitrage，市场是无套利的，一旦出现套利机会价格就会被修正；
- 可以以任意数量做多或做空资产，即可以无限细分资产，可以做空股票，可以借钱；
- Frictionless，不存在交易摩擦。

BS 模型对两种资产价格的假设有：

- 股票无分红(No Dividend)，股价服从参数为常数的几何布朗运动（GBM, Geometric Brownian Motion），即服从随机微分方程（SDE, Stochastic Differential Equation）;
- 利率为常数，债券收益为连续复利。

我们不需要知道如何得出Black Scholes期权定价公式。我们只需要在公式中插入不同的参数，例如看涨/卖出期权，股票价格，执行价格，短期利率，隐含波动率等。现在的问题是我们没有任何方法可以计算隐含波动率。我们只是假设了隐含波动率公式。我们还可以计算看跌期权的价格。例如：苹果股价为130美元。看跌期权的行使价为135美元。有效期为3个月。短期利率为2％。隐含波动率为22％。通过公式可以计算期刊定价为8.214834

Black Scholes期权定价公式很大程度上取决于隐含波动率。隐含波动率是我们所不知道的。因此，实际上我们不能使用此Black Scholes股票期权价格公式。在大多数情况下，我们使用相反的公式。我们在公式中插入股票期权价格并计算隐含波动率。我们可以使用GARCH模型来计算波动率。

## Cox-Ross-Rubinstein期权定价模型

Cox-Ross-Rubinstein公式也称为CRR公式，与Black Scholes股票期权定价公式不同。CRR公式中的基本假设是标的股票价格遵循离散的二项分布。这意味着股票价格在每个时期要么上升一定量，要么下降一定量。二叉树正在重组。这意味着在两个时期内，价格可以先涨后跌，或者在相同的最终价格下涨跌。

这是由于两个公式的数学推导不同。在Black Scholes公式中，我们假设一个连续的随机公式，而在Cox-Ross-Rubinstein公式中，我们假设一个离散的二项式公式。W可以通过减少Cox-Ross-Rubinstein公式中的时间步长来减少价格差异。

## 隐含波动率与BSM

**隐含波动率**，是根据当前期权的市场价格及BS期权定价公式反推得出的。隐含波动率时市场中期权价格所蕴含的波动率，价格和隐含波动率之间是一一对应的，交易者和经纪人经常使用隐含波动率而非价格。
**历史波动率**，是最近20个交易日股价日涨幅的统计标准差。年化波动率就是根号下256乘以每天的标准差，月波动率是20^0.5*日标准差。 比如，若日涨幅统计标准差是1%，那么年波动率就是16%，月波动率就是4.5%。
**未来实际波动率**，是未来的20天的波动率。 比方说，在2020.7.28日收盘时，从2020.7.1-7.28日这20个交易日统计，就是历史波动率。从2020.7.29-8.25日，这20个交易日统计，就是7.28日的股价未来实际波动率。 当然，在7.28日当天，我们是不知道未来20天的实际波动率是多少的。

实际数据，上证50ETF股价日涨幅。 从2020.7.1-7.28日，这20个交易日数据的统计期望值是μ=+0.483%，标准差σ=2.84%；历史波动率就是年化45.44%。 在2020.7.28日，上证50ETF正股价格是3.231元，平值购12月3200的期权价格是0.2167元/份，剩余期限是148天。根据BS期权定价公式反推的隐波大约是年化20%。 从2020.7.29-8.25日，未来实际波动率；统计得到μ=+0.1995%，σ=1.05%；年化实际波动率16.8%。
顺便比较一下，平值沽12月3200，剩余148天，在2020.7.28日的收盘价是0.1901元。
当前2020.11.25日，剩余28天的沽12月3200价格是0.0062元，隐波是22.01%；正股股价是3.499元；而购12月3200已变成深度实值期权，价格是0.3122元，隐波是18.98%，剩余28天。
在2020.6.30日，上证50ETF收盘2.956元，当时剩余176天的购12月3200的价格是0.0460元，隐波是很低的年化13%。从2020.6.1-6.30日，20个交易日的数据统计期望值是0.272%，标准差是0.824%，年化历史波动率也是很低的13.18%。而谁又能料到，未来20天的实际波动率高达45%。
当我们买入认购期权时，除了希望标的正股的股价上涨，还希望隐含波动率也变大。可是，从上面的两个案例，可以看出，隐含波动率的变化也是很难预料的。在第一个案例中，隐含波动率是正常的20%，比历史波动率45%小很多，比未来实际波动率17%差不多。而在紧接着的第二个案例中，隐波是13%，与历史波动率13%相当，但比未来实际波动率45%小很多。
一般，我们把波动率年20%作为一个均值回归线，**如果当前的隐含波动率低于20%，那么买入认购期权是可以期待股价与波动率同时上升的**。而如果期权的隐含波动率已经高达30%以上，就不要再做期权买方了。 比方说，在2020.7.6日，股价继续暴涨一天+9%，收盘3.469元；当时的期权波动率也飙升到40%。理智的讲，不管7.7日第二天股价预计是涨还是跌，都应该清仓认购或认沽期权持仓；因为，波动率会很快就回归到一般正常的20%附近。 实际情形，在7.7日，股价高开低走，当时的剩余170天的平值购12月3400从开盘最高0.5988元大幅下跌至0.2601元，大幅减值一半。 对于当时的剩余30天的短期虚值期权更是一天就大幅减值80%以上；即使后面股价大幅反弹上涨，也得涨5倍才能回本。

"""