# Python Financial Data Analysis (FinanceDA)

## Project Overview

A project that integrates functions such as stock data downloading, analysis, visualization, backtesting, and optimization.

Author: Ye Junjie
Creation Date: May 9, 2021
Last Updated: December 18, 2025

### Core Functional Modules

```python
from financeDA.class_price_list import PriceList # Price list data (stocks, futures, options, etc.), e.g., stock closing prices, stock returns, etc.
from financeDA.class_stock_data import StockData # Stock data including [Close, High, Low, Open, Volume] and key extended indicators [Diff, Signal, Close_Open, Returns, Log_Returns, 42d, 252d, Mov_Vol, etc.]
from financeDA.class_stock_po import StockPO  # Stock portfolio optimization

from financeDA import stat_describe, stat_norm_tests, stat_gen_paths  # Statistical description, normality tests, random path generation
from financeDA import plot_line, plot_candle, plot_hist, plot_QQ, stock_diff, stock_tsa, stock_tests # Visualization: line charts, candlestick charts, histograms, QQ plots, stock return analysis, stock time series analysis, stock statistical tests
from financeDA import ff_reg   # Factor data regression analysis
from financeDA import gbm_mcs_stat, gbm_mcs_dyna, gbm_mcs_amer, option_premium # Option pricing related: GBM model static simulation, dynamic simulation, American option simulation, option premium calculation
from financeDA import bsm_call_imp_vol, bsm_call_value, bsm_vega  # Option pricing related: BSM model implied volatility, option value, option Vega
from financeDA import var_gbm, var_jd, var_diff, var_cva, var_cva_eu  # Value at Risk (VaR) related: GBM model VaR, JD model VaR, incremental VaR, CVA VaR, CVA VaR (European options)
from financeDA import hello_financeda # Print hello_financeda greeting
```

### Acknowledgments

Hilpisch: Python for Financial Data Analysis (Primary reference source for most of the code)
Duan Xiaoshou: Python Quantitative Trading in Depth (Referenced for two components: candlestick charts and simple KNN-based trading strategies)

## Quick Start

### Install the financeDA Package

First, install Python version 3.13.5 or higher (Python 3.14.0 is recommended).
Then download and install the financeDA package:

```bash
pip install financeDA
```

### Single Stock Data Download, Analysis, and Visualization

```python
from financeDA.class_stock_data import StockData # Stock data including [Close, High, Low, Open, Volume] and key extended indicators [Diff, Signal, Close_Open, Returns, Log_Returns, 42d, 252d, Mov_Vol, etc.]
from financeDA import plot_line, plot_candle, plot_hist, plot_QQ, stock_diff, stock_tsa, stock_tests # Visualization: line charts, candlestick charts, histograms, QQ plots, stock return analysis, stock time series analysis, stock statistical tests

df_stock = StockData("BABA", start="2020-01-01", end="2025-12-20", source="yfinance").DF
print(df_stock.head())

plot_line(df_stock["Close"])
plot_candle(df_stock[-21:0])
plot_hist(df_stock["Returns"])
plot_QQ(df_stock["Log_Returns"])

stock_diff(df_stock)
stock_tsa(df_stock)
stock_tests(df_stock)
```

### Portfolio Optimization (Multiple Stocks)

```python
import numpy as np
import pandas as pd
from financeDA.class_stock_data import StockData
from financeDA.class_stock_po import StockPO

stocks = {
    '600031.SH': 'Sany Heavy Industry', 
    '601138.SH': 'Foxconn Industrial Internet', 
    '000768.SZ': 'AVIC Xi\'an Aircraft Industry', 
    '600519.SH': 'Kweichow Moutai'
    }

sdpo = StockPO(stocks, source="tushare", token="your tushare token")

print(sdpo.data.tail())
weights, results = sdpo.po_random()
print(weights, results)

sdpo.po_mc(200000)  # Defaults to 2000 simulations; increase the number for higher accuracy

opts = sdpo.po_max_sharpe()
optv = sdpo.po_min_vol()
print(opts, optv) # Manually calculate portfolio weights for maximum Sharpe ratio and minimum volatility

sdpo.po_ef(trets=np.linspace(0.06, 0.15, 30))  # Manually compute the efficient frontier; adjust trets values for optimal results
sdpo.po_cml() # Manually calculate the maximum Sharpe ratio portfolio under the efficient frontier (Capital Market Line)
```

### Simple Investment Strategy Using Factor Regression

```python
from financeDA  import ff_reg
stocks = {
        "600100.SH": "Tongfang Co., Ltd.",
        "600626.SH": "Shengda Co., Ltd.",
        "000630.SZ": "Tongling Nonferrous Metals",
        "000850.SZ": "Huamao Co., Ltd.",
        "600368.SH": "Wuzhou Transportation",
        "603766.SH": "Loncin General Machinery",
        "600105.SH": "Yongding Co., Ltd.",
        "600603.SH": "Guanghui Logistics",
        "002344.SZ": "Haining Leather City",
        "000407.SZ": "Shengli Co., Ltd.",
        "000883.SZ": "Hubei Energy"
        }
df = ff_reg(stocks=stocks, start_date='2024-10-01', end_date='2025-10-31', mode=5)
print(df)
outfile = "ff_reg.csv"
df.to_csv(outfile)
```

## Project Dependencies

The Python code requires version 3.13.5 or higher (Python 3.14.0 is recommended).

### Dependency Management with uv

Project configuration file: [project.toml]

```bash
uv python install 3.14
uv python pin 3.14
uv sync
uv lock
```

### Dependency Management with Pip

Install required packages (Project dependency file: [requirements.txt])

```bash
pip install numpy pandas matplotlib mplfinance scipy
pip install pandas_datareader yfinance tushare jqdatasdk
pip install scikit-learn statsmodels factor-analyzer

# To speed up installation, append the -i parameter to each command to use domestic pip mirrors. For example:
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

# You can also install dependencies directly using the requirements.txt file
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# For certain Python versions (e.g., Python 3.9.5), if errors occur when running yfinance, you may need to adjust the urllib3 version
pip install urllib3==1.25.11
```

## Mind Map

![Mind Map](fin-data-mind.png)
