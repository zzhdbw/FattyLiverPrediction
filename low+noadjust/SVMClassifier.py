import random
from utils import min_max_data, read_low_file, read_smote_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
from sklearn import svm
random_seed = 7361
random.seed(random_seed)

#数据处理
train_x, train_y, test_x, test_y = read_low_file(train_path = "../low_train.csv", test_path = "../low_test.csv")
train_x, train_y, test_x, test_y = min_max_data(train_x, train_y, test_x, test_y)

# 训练
clf = svm.SVC(probability=True)
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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from sklearn.model_selection import GridSearchCV
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


param = {'criterion': 'gini', 'max_depth': 21, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}

#预测

clf = svm.SVC(kernel = 'linear'
              ,probability=True)  # .SVC（）就是 SVM 的方程，参数 kernel 为线性核函数
# 训练分类器
clf.fit(train_x, train_y)

test_predict_y = clf.predict(test_x)
#roc可视化

test_prob = clf.predict_proba(test_x)[:,1]  #标签为零的概率
fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
auc = metrics.auc(fpr,tpr)
show_roc(fpr, tpr, auc)

#precision_recall可视化
precision, recall, thresholds = precision_recall_curve(test_y, test_prob)
plt.figure(1)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('precision_recall')
plt.show()

#计算指标
test_auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1])
test_acc = accuracy_score(test_y, test_predict_y)
test_precision, test_recall, test_f1 = calculate_PRF(test_y, test_predict_y)

print("test_acc:{: <10.4f}".format(test_acc),
      "test_precision:{: <10.4f}".format(test_precision),
      "test_recall:{: <10.4f}".format(test_recall),
      "test_f1:{: <10.4f}".format(test_f1),
      "test_auc:{: <10.4f}".format(test_auc))

train_predict_y = clf.predict(train_x)
#roc可视化
train_prob = clf.predict_proba(train_x)[:,1]  #标签为零的概率
fpr, tpr, thresholds = metrics.roc_curve(train_y, train_prob)
auc = metrics.auc(fpr,tpr)
show_roc(fpr, tpr, auc)
#计算指标
train_auc = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
train_acc = accuracy_score(train_y, train_predict_y)
train_precision, train_recall, train_f1 = calculate_PRF(train_y, train_predict_y)

print("train_acc:{: <10.4f}".format(train_acc),
      "train_precision:{: <10.4f}".format(train_precision),
      "train_recall:{: <10.4f}".format(train_recall),
      "train_f1:{: <10.4f}".format(train_f1),
      "train_auc:{: <10.4f}".format(train_auc))
exit()


#调参
param_grid = {"n_estimators":np.arange(100,400,30)
              ,"criterion":["gini","entropy"]
              ,"max_depth":np.arange(1,51,5)
              ,"min_samples_leaf":[2,3]
              ,"min_samples_split":[2,3]
              }
best_param = 0
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0

for param in tqdm(list(ParameterGrid(param_grid))):
    # 训练
    rfc = RandomForestClassifier(n_estimators=param["n_estimators"]
                                 ,criterion=param["criterion"]
                                 ,random_state=random_seed
                                 ,max_depth=param["max_depth"]
                                 ,min_samples_leaf=param["min_samples_leaf"]#剪枝
                                 ,min_samples_split=param["min_samples_split"]#剪枝
                                 , n_jobs=-1
                                 )
    clf.fit(train_x, train_y)
    # 评价指标
    test_predict_y = clf.predict(test_x)
    #计算auc
    test_auc = roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])
    test_acc = accuracy_score(test_y, test_predict_y)
    test_precision, test_recall, test_f1 = calculate_PRF(test_y, test_predict_y)

    if(test_auc > best_auc):
        best_param = param
        best_auc = test_auc
        best_acc = test_acc
        best_precision = test_precision
        best_recall = test_recall
        best_f1 = test_f1

print(best_param)
print("test_acc:{: <10.3f}".format(best_acc),
      "test_precision:{: <10.3f}".format(best_precision),
      "test_recall:{: <10.3f}".format(best_recall),
      "test_f1:{: <10.3f}".format(best_f1),
      "test_auc:{: <10.3f}".format(best_auc))



exit()

