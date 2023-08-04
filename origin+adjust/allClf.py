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

def evaluate(clf, train_x, train_y, test_x, test_y):
    # 评价指标
    train_predict_y = clf.predict(train_x)
    train_predict_proba = clf.predict_proba(train_x)[:, 1]
    train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    train_auc = evaluate_auc(train_y, train_predict_proba)

    train_auc95ci = evaluate_auc95ci(clf, train_x, train_y, train_x, train_y, nsamples=200)

    print("train_acc:{: <10.3f}".format(train_acc),
          "train_precision:{: <10.3f}".format(train_precision),
          "train_recall:{: <10.3f}".format(train_recall),
          "train_f1:{: <10.3f}".format(train_f1),
          "train_auc:{: <10.3f}".format(train_auc),
          "train_auc95ci:", train_auc95ci)


    # 混淆矩阵
    p1_a1 = 0
    p1_a0 = 0
    p0_a1 = 0
    p0_a0 = 0


    test_predict_y = clf.predict(test_x)
    print(test_predict_y)
    print([int(i) for i in test_y])

    sum = 0
    for i in range(len(test_y)):
        if(test_y[i] == test_predict_y[i]):
            sum+=1
    print("手算acc:", sum/len(test_predict_y))
    print("sklearn算acc:",accuracy_score(test_y, test_predict_y))
    print("模型算acc:",clf.score(test_x, test_y))
    for i in range(len(test_predict_y)):
        if (test_predict_y[i] == 0 and test_y[i] == 0):
            p0_a0 += 1
        elif (test_predict_y[i] == 0 and test_y[i] == 1):
            p0_a1 += 1
        elif (test_predict_y[i] == 1 and test_y[i] == 0):
            p1_a0 += 1
        elif (test_predict_y[i] == 1 and test_y[i] == 1):
            p1_a1 += 1
    print(p0_a0, p0_a1)
    print(p1_a0, p1_a1)


    test_predict_proba = clf.predict_proba(test_x)[:,1]
    test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    test_auc = evaluate_auc(test_y, test_predict_proba)

    fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)
    test_auc95ci = evaluate_auc95ci(clf, train_x, train_y, test_x, test_y, nsamples=200)

    print("test_acc:{: <10.3f}".format(test_acc),
          "test_precision:{: <10.3f}".format(test_precision),
          "test_recall:{: <10.3f}".format(test_recall),
          "test_f1:{: <10.3f}".format(test_f1),
          "test_auc:{: <10.3f}".format(test_auc),
          "test_auc95ci:", test_auc95ci)

    precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)

    return test_acc, test_precision, test_recall, test_f1, test_auc, test_auc95ci, fpr, tpr, precisions, recalls

def LR(train_x, train_y, test_x, test_y):
    #LR
    print("LR:")
    # {'C': 1.9, 'fit_intercept': 'True', 'penalty': 'l2'}
    LR = LogisticRegression(
        # penalty="l1",
        # C=1.9,
        # fit_intercept=True,
        random_state=random_seed,
        max_iter=10000,
    )
    LR.fit(train_x, train_y)

    # #评价指标
    # test_predict_y = LR.predict(test_x)
    # test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    # test_auc = evaluate_auc(test_y, LR.predict_proba(test_x)[:,1])
    # auc95ci = evaluate_auc95ci(LR, train_x, train_y, test_x, test_y, nsamples=500)
    #
    # print("test_acc:{: <10.3f}".format(test_acc),
    #       "test_precision:{: <10.3f}".format(test_precision),
    #       "test_recall:{: <10.3f}".format(test_recall),
    #       "test_f1:{: <10.3f}".format(test_f1),
    #       "test_auc:{: <10.3f}".format(test_auc),
    #       "test_auc95ci:",auc95ci)
    #
    # #评价指标
    # train_predict_y = LR.predict(train_x)
    # train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    # train_auc = evaluate_auc(train_y, LR.predict_proba(train_x)[:,1])
    #
    # print("train_acc:{: <10.3f}".format(train_acc),
    #       "train_precision:{: <10.3f}".format(train_precision),
    #       "train_recall:{: <10.3f}".format(train_recall),
    #       "train_f1:{: <10.3f}".format(train_f1),
    #       "train_auc:{: <10.3f}".format(train_auc),
    #       "train_auc95ci:",auc95ci)
    return LR
def RFC(train_x, train_y, test_x, test_y):
    #RFC
    print("RFC:")
    #{'criterion': 'entropy', 'max_depth': 12, 'min_samples_leaf': 9, 'min_samples_split': 2, 'n_estimators': 170}
    RFC = RandomForestClassifier(
        n_estimators=20
        # , criterion='entropy'
        # , max_depth=12
        # , min_samples_leaf=9  # 剪枝
        # , min_samples_split=2  # 剪枝
        , random_state=random_seed
        , n_jobs=16
    )
    RFC.fit(train_x, train_y)

    # #评价指标
    # test_predict_y = RFC.predict(test_x)
    # test_predict_proba = RFC.predict_proba(test_x)
    #
    # test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    # test_auc = evaluate_auc(test_y, RFC.predict_proba(test_x)[:,1])
    # auc95ci = evaluate_auc95ci(RFC, train_x, train_y, test_x, test_y, nsamples=500)
    #
    # print("test_acc:{: <10.3f}".format(test_acc),
    #       "test_precision:{: <10.3f}".format(test_precision),
    #       "test_recall:{: <10.3f}".format(test_recall),
    #       "test_f1:{: <10.3f}".format(test_f1),
    #       "test_auc:{: <10.3f}".format(test_auc),
    #       "test_auc95ci:",auc95ci)
    #
    # #评价指标
    # train_predict_y = RFC.predict(train_x)
    # train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    # train_auc = evaluate_auc(train_y, RFC.predict_proba(train_x)[:,1])
    #
    # print("train_acc:{: <10.3f}".format(train_acc),
    #       "train_precision:{: <10.3f}".format(train_precision),
    #       "train_recall:{: <10.3f}".format(train_recall),
    #       "train_f1:{: <10.3f}".format(train_f1),
    #       "train_auc:{: <10.3f}".format(train_auc),
    #       "train_auc95ci:",auc95ci)

    return RFC
    # test_predict_proba = RFC.predict_proba(test_x)[:, 1]
    # fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

    # show_roc(fpr, tpr, test_auc)
    # show_roc(test_y, test_predict_proba)
def SVM(train_x, train_y, test_x, test_y):
    #SVM
    print("SVM:")
    # {'C': 1.5, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
    SVM = svm.SVC(
                  # kernel="linear"
                  # , C=1.5
                  # , degree=3
                  # , gamma="scale"
                  probability=True,
                  random_state=random_seed
                  )

    SVM.fit(train_x, train_y)

    # #评价指标
    # test_predict_y = SVM.predict(test_x)
    # test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    # test_auc = evaluate_auc(test_y, SVM.predict_proba(test_x)[:,1])
    # auc95ci = evaluate_auc95ci(SVM, train_x, train_y, test_x, test_y, nsamples=500)
    #
    # print("test_acc:{: <10.3f}".format(test_acc),
    #       "test_precision:{: <10.3f}".format(test_precision),
    #       "test_recall:{: <10.3f}".format(test_recall),
    #       "test_f1:{: <10.3f}".format(test_f1),
    #       "test_auc:{: <10.3f}".format(test_auc),
    #       "test_auc95ci:",auc95ci)
    #
    # #评价指标
    # train_predict_y = SVM.predict(train_x)
    # train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    # train_auc = evaluate_auc(train_y, SVM.predict_proba(train_x)[:,1])
    #
    # print("train_acc:{: <10.3f}".format(train_acc),
    #       "train_precision:{: <10.3f}".format(train_precision),
    #       "train_recall:{: <10.3f}".format(train_recall),
    #       "train_f1:{: <10.3f}".format(train_f1),
    #       "train_auc:{: <10.3f}".format(train_auc),
    #       "train_auc95ci:",auc95ci)

    return SVM
def GBC(train_x, train_y, test_x, test_y):

    #GBC
    print("GBC:")
    # {'criterion': 'friedman_mse', 'learning_rate': 0.15, 'max_depth': 1, 'min_samples_leaf': 11, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 0.8}
    GBC = GradientBoostingClassifier(
                                        n_estimators=30
                                        #  ,learning_rate=0.15
                                        #  , subsample=0.8
                                        #  , criterion="friedman_mse"
                                        #  , max_depth=1
                                        #  , min_samples_leaf=11  # 剪枝
                                        #  , min_samples_split=2  # 剪枝
                                         ,random_state=random_seed
                                         )

    GBC.fit(train_x, train_y)

    # #评价指标
    # test_predict_y = GBC.predict(test_x)
    # test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    # test_auc = evaluate_auc(test_y, GBC.predict_proba(test_x)[:,1])
    # auc95ci = evaluate_auc95ci(GBC, train_x, train_y, test_x, test_y, nsamples=500)
    #
    # print("test_acc:{: <10.3f}".format(test_acc),
    #       "test_precision:{: <10.3f}".format(test_precision),
    #       "test_recall:{: <10.3f}".format(test_recall),
    #       "test_f1:{: <10.3f}".format(test_f1),
    #       "test_auc:{: <10.3f}".format(test_auc),
    #       "test_auc95ci:",auc95ci)
    #
    # #评价指标
    # train_predict_y = GBC.predict(train_x)
    # train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    # train_auc = evaluate_auc(train_y, GBC.predict_proba(train_x)[:,1])
    #
    # print("train_acc:{: <10.3f}".format(train_acc),
    #       "train_precision:{: <10.3f}".format(train_precision),
    #       "train_recall:{: <10.3f}".format(train_recall),
    #       "train_f1:{: <10.3f}".format(train_f1),
    #       "train_auc:{: <10.3f}".format(train_auc),
    #       "train_auc95ci:",auc95ci)
    return GBC
def XGB(train_x, train_y, test_x, test_y):
    #XGB
    print("XGB:")
    # {'alpha': 0.005, 'booster': 'gbtree', 'colsample_bytree': 0.9,
    # 'gamma': 0.01, 'learning_rate': 0.1, 'max_depth': 1,
    #  'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 0.01, 'subsample': 0.9}

    XGB = XGBC(
        n_estimators=200
        , booster='gbtree'
        , learning_rate=0.15
        , gamma=0.01
        , alpha=0.1
        , reg_lambda=0.02
        , max_depth=1
        , min_child_weight=3
        , subsample=0.8
        , colsample_bytree=0.8
        , objective="binary:logistic"
        , eval_metric="auc"
        , seed=random_seed
        # , use_label_encoder=False
        , n_jobs=16
    )
    XGB.fit(train_x, train_y)

    # #评价指标
    # test_predict_y = XGB.predict(test_x)
    # test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
    # test_auc = evaluate_auc(test_y, XGB.predict_proba(test_x)[:,1])
    # auc95ci = evaluate_auc95ci(XGB, train_x, train_y, test_x, test_y, nsamples=500)
    #
    # print("test_acc:{: <10.3f}".format(test_acc),
    #       "test_precision:{: <10.3f}".format(test_precision),
    #       "test_recall:{: <10.3f}".format(test_recall),
    #       "test_f1:{: <10.3f}".format(test_f1),
    #       "test_auc:{: <10.3f}".format(test_auc),
    #       "test_auc95ci:",auc95ci)
    #
    # #评价指标
    # train_predict_y = XGB.predict(train_x)
    # train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
    # train_auc = evaluate_auc(train_y, XGB.predict_proba(train_x)[:,1])
    #
    # print("train_acc:{: <10.3f}".format(train_acc),
    #       "train_precision:{: <10.3f}".format(train_precision),
    #       "train_recall:{: <10.3f}".format(train_recall),
    #       "train_f1:{: <10.3f}".format(train_f1),
    #       "train_auc:{: <10.3f}".format(train_auc),
    #       "train_auc95ci:",auc95ci)

    return XGB
def Improved_XGB(train_x, train_y, test_x, test_y):
    #XGB
    print("ImprovedXGB:")

    #origin
    # {'alpha': 0.005, 'booster': 'gbtree', 'colsample_bytree': 0.9,
    # 'gamma': 0.01, 'learning_rate': 0.1, 'max_depth': 1,
    #  'min_child_weight': 1, 'n_estimators': 200, 'reg_lambda': 0.01, 'subsample': 0.9}

    #smote
    # {'alpha': 0.005, 'booster': 'gbtree', 'colsample_bytree': 0.8,
    #  'gamma': 0.01, 'learning_rate': 0.15, 'max_depth': 1,
    #  'min_child_weight': 3, 'n_estimators': 200, 'reg_lambda': 0.02, 'subsample': 0.8}
    #

    XGB = XGBC(
            n_estimators=200
            , booster='gbtree'
            , learning_rate=0.15
            , gamma=0.01
            , alpha=0.005
            , reg_lambda=0.02
            , max_depth=1
            , min_child_weight=3
            , subsample=0.8
            , colsample_bytree=0.8
            , objective="binary:logistic"
            , eval_metric="auc"
            , seed=random_seed
            # , use_label_encoder=False
            , n_jobs=16
        )

    XGB = XGB.fit(train_x, train_y)

    return XGB


if __name__ == '__main__':
    # 数据处理
    # train_x, train_y, test_x, test_y = read_file(train_path="../训练集.csv", test_path="../测试集.csv")
    # train_x, train_y, test_x, test_y = read_file(train_path="../data/origin/训练集.csv", test_path="../data/origin/测试集.csv")

    train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/训练集.csv",
                                                 test_path="../data/shuffle/测试集.csv",
                                                 encoding="utf8")
    #
    # train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/smote_train.csv",
    #                                              test_path="../data/shuffle/smote_test.csv",
    #                                              encoding="utf8")


    #绘图头####################################################################
    plt.figure("ROC Curve")
    ROC = plt.gca()
    lw = 2

    plt.figure("P-R Curve")
    PR =  plt.gca()
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # RFC
    RFC = RandomForestClassifier(
        n_estimators=20
        # , criterion='entropy'
        # , max_depth=12
        # , min_samples_leaf=9  # 剪枝
        # , min_samples_split=2  # 剪枝
        , random_state=random_seed
        , n_jobs=16
    )
    RFC.fit(train_x, train_y)

    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y, test_x, test_y)

    plt.sca(ROC)
    ROC.plot(fpr,
             tpr,
             lw=lw,
             label='RF ROC curve (AUC=%0.3f)' % test_auc
             )

    plt.sca(PR)
    plt.plot(recalls, precisions, label='RF PR curve')

    # GBC
    clf = GBC(train_x, train_y, test_x, test_y)
    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y,
                                                                                           test_x, test_y)
    plt.sca(ROC)
    plt.plot(fpr,
             tpr,
             lw=lw,
             label='GBM ROC curve (AUC=%0.3f)' % test_auc
             )
    plt.sca(PR)
    plt.plot(recalls, precisions, label='GBM PR curve')

    # XGB
    clf = XGB(train_x, train_y, test_x, test_y)
    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y,
                                                                                           test_x, test_y)
    plt.sca(ROC)
    plt.plot(fpr,
             tpr,
             lw=lw,
             label='XGB ROC curve (AUC=%0.3f)' % test_auc
             )
    plt.sca(PR)
    plt.plot(recalls, precisions, label='XGB PR curve')
    train_x, train_y, test_x, test_y = min_max_data(train_x, train_y, test_x, test_y)

    # LR
    clf = LR(train_x, train_y, test_x, test_y)
    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y,
                                                                                           test_x, test_y)
    plt.sca(ROC)
    plt.plot(fpr,
             tpr,
             lw=lw,
             label='LR ROC curve (AUC=%0.3f)' % test_auc
             )
    plt.sca(PR)
    plt.plot(recalls, precisions, label='LR PR curve')
    clf = SVM(train_x, train_y, test_x, test_y)
    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y,
                                                                                           test_x, test_y)
    plt.sca(ROC)
    plt.plot(fpr,
             tpr,
             lw=lw,
             label='SVM ROC curve (AUC=%0.3f)' % test_auc
             )
    plt.sca(PR)
    plt.plot(recalls, precisions, label='SVM PR curve')


    #Improved_XGB
    clf = Improved_XGB(train_x, train_y, test_x, test_y)
    test_acc, test_precision, test_recall, test_f1, test_auc, auc95ci, fpr, tpr, precisions, recalls = evaluate(clf, train_x, train_y, test_x, test_y)


    #混淆矩阵
    p1_a1 = 0
    p1_a0 = 0
    p0_a1 = 0
    p0_a0 = 0
    test_predict_y = clf.predict(test_x)
    for i in range(len(test_predict_y)):
        if (test_predict_y[i] == 0 and test_y[i] == 0):
            p0_a0 += 1
        elif (test_predict_y[i] == 0 and test_y[i] == 1):
            p0_a1 += 1
        elif (test_predict_y[i] == 1 and test_y[i] == 0):
            p1_a0 += 1
        elif (test_predict_y[i] == 1 and test_y[i] == 1):
            p1_a1 += 1
    print(p0_a0, p0_a1)
    print(p1_a0, p1_a1)


    plt.sca(ROC)
    plt.plot(fpr,
             tpr,
             lw=lw,
             label='ImprovedXGB ROC curve (AUC=%0.3f)' % test_auc
             )

    #特征重要性
    features = list(test_x.columns)
    print(clf.feature_importances_)
    print(features)

    plt.sca(PR)
    plt.plot(recalls, precisions, label='Improved_XGB PR curve')
    plt.legend()


    plt.sca(ROC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
