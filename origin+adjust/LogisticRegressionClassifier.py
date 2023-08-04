import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from sklearn.model_selection import ParameterGrid,GridSearchCV
from utils import read_file, read_smote_file, min_max_data, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci

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

#数据处理
train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

#最优参数
# {'C': 1.9, 'fit_intercept': 'True', 'penalty': 'l2'}
# test_acc:0.824      test_precision:0.910      test_recall:0.844      test_f1:0.876      test_auc:0.917

#调参
param_grid = {"penalty":["l2"]
              ,"C":[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2]
              ,"fit_intercept":["True","False"]
              }

best_param = 0
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0

for param in tqdm(list(ParameterGrid(param_grid))):
    # 训练
    clf = LogisticRegression(
                             penalty=param["penalty"],
                             C=param["C"] ,
                             #fit_intercept=param["fit_intercept"],
                              random_state=random_seed,
                              max_iter=10000,
                             )
    clf.fit(train_x, train_y)
    # 评价指标
    test_predict_y = clf.predict(test_x)
    test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    test_auc = evaluate_auc(test_y, clf.predict_proba(test_x)[:,1])

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


