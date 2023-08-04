import random
from utils import read_low_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
from xgboost import XGBClassifier as XGBC
random_seed = 7361
random.seed(random_seed)

#数据处理
train_x, train_y, test_x, test_y = read_low_file(train_path = "../low_train.csv", test_path = "../low_test.csv")

# 训练
clf = XGBC(
      # use_label_encoder=False
      n_estimators=200
    , n_jobs=-1
    , seed=random_seed
)

clf.fit(train_x, train_y)

#预测train
train_predict_y = clf.predict(train_x)
train_predict_proba = clf.predict_proba(train_x)[:,1]

#评价指标
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc_score = evaluate_auc(train_y, train_predict_proba)
auc95ci = evaluate_auc95ci(clf, train_x, train_y, test_x, test_y, nsamples=500)

print("train_acc:{: <10.4f}".format(train_acc)
      ,"train_precision:{: <10.4f}".format(train_precision)
      ,"train_recall:{: <10.4f}".format(train_recall)
      ,"train_f1:{: <10.4f}".format(train_f1)
      ,"train_auc:{: <10.4f}".format(train_auc_score)
      ,"train_auc95ci:", str(auc95ci)
      )

#预测test
test_predict_y = clf.predict(test_x)
test_predict_proba = clf.predict_proba(test_x)[:,1]

#评价指标
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc_score = evaluate_auc(test_y, test_predict_proba)

print("test_acc:{: <10.4f}".format(test_acc)
      ,"test_precision:{: <10.4f}".format(test_precision)
      ,"test_recall:{: <10.4f}".format(test_recall)
      ,"test_f1:{: <10.4f}".format(test_f1)
      ,"test_auc:{: <10.4f}".format(test_auc_score)
      ,"test_auc95ci:", str(auc95ci)
      )
exit()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from tqdm import tqdm
import random
random_seed = 7361
random.seed(random_seed)

def show_roc(fpr, tpr, auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def calculate_PRF(y, y_predict):
    test_precision = precision_score(y, y_predict, average='binary')
    test_recall = recall_score(y, y_predict, average='binary')
    test_f1 = f1_score(y, y_predict, average='binary')
    return  test_precision, test_recall, test_f1
from utils import read_file
#数据处理
train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

reg = XGBC(
    # max_depth=6,#树的最大深度
    learning_rate=0.1#学习率
    ,n_estimators=100#多少棵树
    # ,silent=False#是否输出过程
    ,objective='binary:logistic'#最小化的损失函数。
    ,booster='gbtree'
    ,gamma=0
    ,min_child_weight=1
    ,subsample=0.8#观测的子样本的比率，即对总体进行随机抽样的比例。默认为1
    ,colsample_bytree=0.8#用于构造每棵树时变量的子样本比率.即特征抽样。默认为1。
    ,reg_alpha=0
    ,n_jobs=-1
    ,seed=888)

reg.fit(train_x, train_y)

print(reg.get_params())
print(reg.score(test_x,test_y))
print(accuracy_score(test_y, reg.predict(test_x)))

test_prob = reg.predict_proba(test_x)[:,1]  #标签为零的概率
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
auc = metrics.auc(fpr,tpr)
show_roc(fpr, tpr, auc)
print(auc)
exit()

