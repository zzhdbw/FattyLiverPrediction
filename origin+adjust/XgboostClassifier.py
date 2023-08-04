import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import random
from utils import read_file, read_smote_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
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
# train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/训练集.csv",
#                                              test_path="../data/shuffle/测试集.csv",
#                                              encoding="utf8")
train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/smote_train.csv",
                                                 test_path="../data/shuffle/smote_test.csv",
                                                 encoding="utf8")
# #调参
param_grid = {
    "n_estimators": [200,250,300],
    "booster": ["gbtree", "gblinear"],
    "learning_rate": [0.05, 0.1, 0.15],
    "gamma": [0.01,0.005,0.1, 0.2],
    "alpha": [0.005, 0.01],
    "reg_lambda": [0.01, 0.02],
    "max_depth": np.arange(1, 5, 1),
    "min_child_weight": np.arange(1, 4, 1),
    "subsample": [0.8, 0.9, 0.95],
    "colsample_bytree": [0.8, 0.9, 0.95],
}

best_param = 0
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0
best_clf = 0

for param in tqdm(list(ParameterGrid(param_grid))):
    # 训练
    clf = XGBC(
        n_estimators=param["n_estimators"]
        , booster=param["booster"]
        , learning_rate=param["learning_rate"]
        , gamma=param["gamma"]
        , alpha=param["alpha"]
        , reg_lambda=param["reg_lambda"]
        , max_depth=param["max_depth"]
        , min_child_weight=param["min_child_weight"]
        , subsample=param["subsample"]
        , colsample_bytree=param["colsample_bytree"]
        , objective="binary:logistic"
        , eval_metric="auc"
        , seed=random_seed
        # , use_label_encoder=False
        , n_jobs=16
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
        best_clf = clf

print(best_param)
# print("test_acc:{: <10.3f}".format(best_acc),
#       "test_precision:{: <10.3f}".format(best_precision),
#       "test_recall:{: <10.3f}".format(best_recall),
#       "test_f1:{: <10.3f}".format(best_f1),
#       "test_auc:{: <10.3f}".format(best_auc))

#评价指标
test_predict_y = best_clf.predict(test_x)
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, best_clf.predict_proba(test_x)[:,1])
auc95ci = evaluate_auc95ci(best_clf, train_x, train_y, test_x, test_y, nsamples=500)

print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:",auc95ci)

#评价指标
train_predict_y = best_clf.predict(train_x)
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, best_clf.predict_proba(train_x)[:,1])

print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      "train_auc95ci:",auc95ci)

