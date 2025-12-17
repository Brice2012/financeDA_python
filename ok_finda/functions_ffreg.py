# 使用五因子模型做量化分析（多支股票）
# 完整代码（可复制到一个py文件中运行）

import token
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from .functions_stock import daily_return_ratio, daily_return_ratio_log
from .functions_stock import sum_return_ratio, max_draw_down, sharpe_ratio, information_ratio, treynor_ratio
from .functions_stock import get_stock_data_yf, get_stock_data_ts

plt.rcParams ['font.sans-serif'] ='SimHei'  #显示中文
plt.rcParams ['axes.unicode_minus']=False   #显示负号

def deal(code:str, start_date:str, end_date:str, three_factors:pd.DataFrame, mode:int=3, source:str="yfinance", token:str=None): 
    result_dict =  {'股票代码':code}
    if source=="tushare" and token is not None:
        stock_data = get_stock_data_ts(code, start_date, end_date, token=token)
    else:
        stock_data = get_stock_data_yf(code, start_date, end_date)
    
    result_dict["实际总收益率"]=sum_return_ratio(stock_data['Close'])
    result_dict["最大回测率"]=max_draw_down(stock_data['Close'])
    result_dict["夏普比率"]=sharpe_ratio(stock_data['Close'])
    result_dict["信息比率"]=information_ratio(stock_data['Close'])
    result_dict["特雷诺比率"]=treynor_ratio(stock_data['Close'], beta=1)
    
    stock_data['Returns'] = daily_return_ratio(stock_data['Close']) #['收益率']
    
    zgpa_threefactor = pd.merge(three_factors, stock_data[['Returns']],left_index=True, right_index=True)    
    if mode==5:
        # print("mode=5")
        result = smf.ols('Returns ~ mkt_rf + smb + hml + rmw + cma', data=zgpa_threefactor).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        # print(result.summary())
        betas=result.params
        # print(betas)
        result_dict['阿尔法(alpha)'] = betas['Intercept']
        result_dict['市场因子MKT'] = betas['mkt_rf']
        result_dict['规模因子SMB'] = betas['smb']
        result_dict['价值因子HML'] = betas['hml'] 
        result_dict['盈利能力因子RMW'] = betas['rmw']
        result_dict['投资模式因子CMA'] = betas['cma']
        # print(result_dict)
        return result_dict
    else:
        # print("mode=3")
        result = smf.ols('Returns ~ mkt_rf + smb + hml', data=zgpa_threefactor).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        # print(result.summary())
        betas=result.params
        # print(betas)
        result_dict['阿尔法(alpha)'] = betas['Intercept']
        result_dict['市场因子MKT'] = betas['mkt_rf']
        result_dict['规模因子SMB'] = betas['smb']
        result_dict['价值因子HML'] = betas['hml'] 
        # print(result_dict)
        return result_dict
        
def ff_reg ( codes:dict,  start_date:str, end_date:str, mode:int=3, factors: str = None):
    from importlib.resources import files
    from importlib.resources.abc import Traversable
    # 读取因子数据
    if factors:
        three_factors = pd.read_csv(factors)[['trddy','mkt_rf','smb','hml','rmw','cma']].rename(columns={'trddy':'Date'}).set_index('Date')
    else:
        # 定位本地factors—CSV文件（适配安装后的路径）
        data_dir: Traversable = files("ok_finda") / "data"
        csv_path: Traversable = data_dir / "ff_5factors_daily.csv"
        three_factors = pd.read_csv(csv_path)[['trddy','mkt_rf','smb','hml','rmw','cma']].rename(columns={'trddy':'Date'}).set_index('Date')
            
    three_factors = three_factors.loc[start_date:end_date,:]
    three_factors.index = pd.to_datetime(three_factors.index)
    # print(three_factors.head())
    # print(three_factors.tail())

    code_list=list(codes.keys())
    df_results=pd.DataFrame()
    for code in code_list:
        try:
            r_dict = deal(code=code,mode=mode,start_date=start_date,end_date=end_date,three_factors=three_factors)
            r_dict['股票名称']=codes[code]
            print(r_dict)
            r_df = pd.DataFrame(r_dict,index=[0])
            df_results=pd.concat([df_results,r_df],axis=0,ignore_index=True)
            print(f'股票{codes[code]}({code})处理完成:{r_dict}')
        except Exception as e:      
            print(f"处理股票{code}时出错：{e}")
            pass  
    # df_results=df_results.sort_values(by='阿尔法(alpha)',ascending=False)
    return(df_results)

#%%
if __name__ == '__main__':
    code_name={
            "600100.SH": "同方股份",
            "600626.SH": "申达股份",
            "000630.SZ": "铜陵有色",
            "000850.SZ": "华茂股份",
            "600368.SH": "五洲交通",
            "603766.SH": "隆鑫通用",
            "600105.SH": "永鼎股份",
            "600603.SH": "广汇物流",
            "002344.SZ": "海宁皮城",
            "000407.SZ": "胜利股份",
            "000883.SZ": "湖北能源"
            }
    df = ff_reg(codes=code_name, start_date='2024-10-01', end_date='2025-10-31', mode=5, factors="./data/ff_5factors_daily_CUFE.csv")
    print(df)
    outfile = "ff_reg.csv"
    df.to_csv(outfile)
