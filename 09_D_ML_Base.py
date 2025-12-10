# ######################################
# python 金融数据分析
# Author: 叶俊杰
# Date: 2025-11-28
# #######################################

#%%
### ch9. 机器学习概述
######################

# 9.1 机器学在金融中的应用
# 9.2 ML类型、过程及常用模型
# 9.3 人工智能在金融中的应用

#%%
#导入需要的包
#使用pip安装包: pip install scikit_learn statsmodels
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

import statsmodels.formula.api as smf #导入统计模型包
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression #导入机器学习库中的线性回归模块
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取  

#%%
# 9.1 机器学习在金融中的应用
# ####################################

# ML应用：
# 1. 股票预测：根据历史数据预测股票价格
# 2. 信用风险评估：根据客户的历史交易记录和风险承受能力评估客户的信用风险
# 3. 欺诈检测：识别可能的欺诈交易或活动
# 4. 客户服务：自动回答客户问题，提供个性化建议
# 5. 推荐系统：根据用户的历史行为和偏好推荐商品或服务


#%%
# 9.2 类型、过程及常用模型
# #########################

# 类型：回归-分类-聚类-降维
# 过程：训练-测试-评估/预测

# ML模型：LR,KNN,FCA等

print('本课程学习的模型：LR,KNN,FCA等')

#%%
# 9.3 人工智能在金融中的应用
# ####################################

# AIGC：文本分析，舆情分析，情感分析等

#%%
#############END################
