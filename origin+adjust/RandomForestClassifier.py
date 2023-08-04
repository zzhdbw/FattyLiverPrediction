import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import random
from utils import read_file, read_low_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
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

# #数据处理
# train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

train_x, train_y, test_x, test_y = read_file(train_path = "../low_train.csv", test_path = "../low_test.csv")


# {'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 9, 'min_samples_split': 2, 'n_estimators': 170}
# test_acc:0.839      test_precision:0.810      test_recall:0.885      test_f1:0.846      test_auc:0.944

#调参
param_grid = {"n_estimators":np.arange(160,201,10)
              ,"criterion":["gini","entropy"]
              ,"max_depth":np.arange(12,21,1)
              ,"min_samples_leaf":np.arange(1,11,1)
              ,"min_samples_split":np.arange(2,11,1)
              }

best_param = 0
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0

for param in tqdm(list(ParameterGrid(param_grid))):
    # 训练
    clf = RandomForestClassifier(n_estimators=param["n_estimators"]
                                 ,criterion=param["criterion"]
                                 ,random_state=random_seed
                                 ,max_depth=param["max_depth"]
                                 ,min_samples_leaf=param["min_samples_leaf"]#剪枝
                                 ,min_samples_split=param["min_samples_split"]#剪枝
                                 ,n_jobs=16
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


