# findaok/__init__.py
__version__ = "0.1.6"  # 版本号（与setup.cfg一致）

from .main import help
from .functions_stock import load_stock_csv, get_stock_data_ts, get_stock_data_yf
from .functions_stock import set_stock_excol, daily_return_ratio, daily_return_ratio_log
from .functions_stock import sum_return_ratio, max_draw_down, sharpe_ratio, information_ratio, treynor_ratio
from .functions_stat import stat_describe, normality_tests, gen_paths
from .functions_bsm import bsm_call_imp_vol, bsm_call_value, bsm_vega
from .functions_single import stock_tsa, stock_candle, stock_test
from .functions_po import port_ret_vol_random, port_ret_vol_montecarlo, opts_max_sharpe, opts_min_vol, po_efficient_frontier, po_cap_market_line
from .functions_val import gbm_mcs_stat, gbm_mcs_dyna, gbm_mcs_amer, opt_premium_euro_amer
from .functions_var import var_gbm, var_jd, var_diff, var_cva, var_cva_eu
from .functions_ffreg import ff_reg

__all__ = ["help", "load_stock_csv", "get_stock_data_ts", "get_stock_data_yf", "set_stock_excol", "daily_return_ratio", "daily_return_ratio_log", "sum_return_ratio", "max_draw_down", "sharpe_ratio", "information_ratio", "treynor_ratio", "stat_describe", "normality_tests", "gen_paths", "bsm_call_imp_vol", "bsm_call_value", "bsm_vega", "stock_tsa", "stock_candle", "stock_test", "port_ret_vol_random", "port_ret_vol_montecarlo", "opts_max_sharpe", "opts_min_vol", "po_efficient_frontier", "po_cap_market_line", "gbm_mcs_stat", "gbm_mcs_dyna", "gbm_mcs_amer", "opt_premium_euro_amer", "var_gbm", "var_jd", "var_diff", "var_cva", "var_cva_eu", "ff_reg"]  # 导出的公开接口