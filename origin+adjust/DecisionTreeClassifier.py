import pandas as pd
import sklearn
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
import random
random_seed = 7361
random.seed(random_seed)
from utils import read_file

#数据处理
train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

#训练
clf = tree.DecisionTreeClassifier(criterion='gini'#'entropy'#信息熵  'gini'#基尼系数
                                  ,random_state=random_seed
                                  ,splitter="random"# "best"选最好的    "random"随机选 能减缓过拟合
                                  ,max_depth=10#剪枝 最大深度
                                  ,min_samples_leaf=5#剪枝
                                  ,min_samples_split=10#剪枝
                                  )
clf.fit(train_x, train_y)

#评价指标
test_predict_y = clf.predict(test_x)
test_acc = accuracy_score(test_y, test_predict_y)
test_precision = precision_score(test_y, test_predict_y, average='binary')
test_recall = recall_score(test_y, test_predict_y, average='binary')
test_f1 = f1_score(test_y, test_predict_y, average='binary')
print("test_acc:{: <10.3f}".format(test_acc),"test_precision:{: <10.3f}".format(test_precision),"test_recall:{: <10.3f}".format(test_recall),"test_f1:{: <10.3f}".format(test_f1))

train_predict_y = clf.predict(train_x)
train_acc = accuracy_score(train_y, train_predict_y)
train_precision = precision_score(train_y, train_predict_y, average='binary')
train_recall = recall_score(train_y, train_predict_y, average='binary')
train_f1 = f1_score(train_y, train_predict_y, average='binary')
print("train_acc:{: <10.3f}".format(train_acc),"train_precision:{: <10.3f}".format(train_precision),"train_recall:{: <10.3f}".format(train_recall),"train_f1:{: <10.3f}".format(train_f1))

# test_acc:0.840      test_precision:0.903      test_recall:0.875      test_f1:0.889
# train_acc:0.817      train_precision:0.804      train_recall:0.897      train_f1:0.848
# 0.8089285714285714


# # print(roc_auc_score(test_y, test_predict_y))
fpr, tpr, thresholds = roc_curve(test_y, test_predict_y)
auc = sklearn.metrics.auc(fpr, tpr)
print(auc)
# # precision_recall_curve
#
# import matplotlib.pyplot as plt
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
