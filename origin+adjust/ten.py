import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from utils import read_file, show_roc, min_max_data, read_smote_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
random_seed = 7361
random.seed(random_seed)




from sklearn.model_selection import KFold




# 读取数据
train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/训练集.csv",
                                             test_path="../data/shuffle/测试集.csv",
                                             encoding="utf8")





KF = KFold(n_splits=10,shuffle=False)

acc_sum = 0
auc_sum = 0
for train_index,test_index in KF.split(train_x):
    XGB = XGBC(
        n_estimators=200
        # , booster='gbtree'
        , learning_rate=0.01
        # , gamma=0.01
        # , alpha=0.1
        # , reg_lambda=0.02
        # , max_depth=1
        # , min_child_weight=3
        , subsample=0.9
        , colsample_bytree=0.8
        , objective="binary:logistic"
        , eval_metric="auc"
        , seed=random_seed
        # , use_label_encoder=False
        , n_jobs=16
    )
    train_x_fold = (train_x.iloc[train_index])
    train_y_fold = (train_y.iloc[train_index])

    test_x_fold = (train_x.iloc[test_index])
    test_y_fold = (train_y.iloc[test_index])

    XGB.fit(train_x_fold, train_y_fold)

    # acc = XGB.score(test_x_fold, test_y_fold)
    # print(acc)
    # acc_sum +=acc

    test_predict_proba = XGB.predict_proba(test_x_fold)[:, 1]
    train_auc = evaluate_auc(test_y_fold, test_predict_proba)
    auc_sum += train_auc
    print(train_auc)
# print(acc_sum/10)
print(auc_sum/10)





