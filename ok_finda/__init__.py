# findaok/__init__.py
__version__ = "0.1.0"  # 版本号（与setup.cfg一致）

from .main import hello_world
from .functions_stock import load_stock_data, load_stock_csv, set_stock_excol, daily_return_ratio, daily_return_ratio_log, sum_return_ratio, max_draw_down, sharpe_ratio, information_ratio, treynor_ratio

__all__ = ["hello_world", "load_stock_data", "load_stock_csv", "set_stock_excol", "daily_return_ratio", "daily_return_ratio_log", "sum_return_ratio", "max_draw_down", "sharpe_ratio", "information_ratio", "treynor_ratio"]  # 导出的公开接口