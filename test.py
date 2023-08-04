# coding:utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

#绘制特征重要性图

# -*- 原始数据 -*-
Feature_importances = [0.05292855, 0.26349786 ,0.05554461 ,0.13898556 ,0.17149189, 0.08890491
 ,0.08265503 ,0.14599156]

fea_label = ['Age', 'VAI', 'Hdl_c', 'Abdominal', 'BMI', 'Alt/Ast', 'Alt', 'Gtg']

Feature_importances = [round(x, 4) for x in Feature_importances]
F2 = pd.Series(Feature_importances, index=fea_label)
F2 = F2.sort_values(ascending=True)
f_index = F2.index
f_values = F2.values

# -*-输出 -*- # 
print('f_index:', f_index)
print('f_values:', f_values)
#####################################
x_index = list(range(0, 8))
x_index = [x / 8 for x in x_index]
# plt.rcParams['figure.figsize'] = (10, 10)

plt.barh(x_index, f_values, height=0.1, align="center", color='#1f77b4', tick_label=f_index)

plt.xlabel('Importances')
plt.ylabel('Features')
plt.show()
