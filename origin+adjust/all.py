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

#读取数据
train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/训练集.csv",
                                             test_path="../data/shuffle/测试集.csv",
                                             encoding="utf8")
# train_x, train_y, test_x, test_y = read_file(train_path="../data/shuffle/smote_train.csv",
#                                              test_path="../data/shuffle/smote_test.csv",
#                                              encoding="utf8")
# 绘图头####################################################################
plt.figure("ROC Curve")
ROC = plt.gca()
lw = 2

plt.figure("P-R Curve")
PR = plt.gca()
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')


print("LR============================================================================================================================")

LR = LogisticRegression(
    # penalty="l2",
    # C=2,
    # fit_intercept=True,
    # class_weight = 0.5,
    random_state=random_seed,
    max_iter=80,
)
LR.fit(train_x, train_y)
#train预测
train_predict_y = LR.predict(train_x)
train_predict_proba = LR.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(LR, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = LR.predict(test_x)
test_predict_proba = LR.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(LR, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0],test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", LR.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

plt.sca(ROC)
ROC.plot(fpr,tpr,lw=lw,label='LR ROC curve (AUC=%0.3f)' % test_auc)

plt.sca(PR)
plt.plot(recalls, precisions, label='LR PR curve')
print("==============================================================================================================================")


print("RFC===========================================================================================================================")
# RFC
RFC = RandomForestClassifier(
    n_estimators=20
    , random_state=random_seed
    , n_jobs=16
)
RFC.fit(train_x, train_y)

#train预测
train_predict_y = RFC.predict(train_x)
train_predict_proba = RFC.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(RFC, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = RFC.predict(test_x)
test_predict_proba = RFC.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(RFC, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0], test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", RFC.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

plt.sca(ROC)
ROC.plot(fpr,
         tpr,
         lw=lw,
         label='RF ROC curve (AUC=%0.3f)' % test_auc
         )

plt.sca(PR)
plt.plot(recalls, precisions, label='RF PR curve')

print("==============================================================================================================================")
print("XGB===========================================================================================================================")
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
XGB.fit(train_x, train_y)
#train预测
train_predict_y = XGB.predict(train_x)
train_predict_proba = XGB.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(XGB, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = XGB.predict(test_x)
test_predict_proba = XGB.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(XGB, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0],test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", XGB.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

plt.sca(ROC)
ROC.plot(fpr,tpr,lw=lw,label='XGB ROC curve (AUC=%0.3f)' % test_auc)

plt.sca(PR)
plt.plot(recalls, precisions, label='XGB PR curve')
print("==============================================================================================================================")
print("GBM============================================================================================================================")

GBM = GradientBoostingClassifier(
    n_estimators=13
    #  ,learning_rate=0.15
    #  , subsample=0.8
    #  , criterion="friedman_mse"
    #  , max_depth=1
    #  , min_samples_leaf=11  # 剪枝
    #  , min_samples_split=2  # 剪枝
    , random_state=random_seed
)

GBM.fit(train_x, train_y)
#train预测
train_predict_y = GBM.predict(train_x)
train_predict_proba = GBM.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(GBM, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = GBM.predict(test_x)
test_predict_proba = GBM.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(GBM, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0],test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", GBM.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

plt.sca(ROC)
ROC.plot(fpr,tpr,lw=lw,label='GBM ROC curve (AUC=%0.3f)' % test_auc)

plt.sca(PR)
plt.plot(recalls, precisions, label='GBM PR curve')
print("==============================================================================================================================")

print("SVM============================================================================================================================")

SVM = svm.SVC(
    # kernel="linear"
    # , C=1.5
    # , degree=3
    # , gamma="scale"
    probability=True,
    random_state=random_seed
)

SVM.fit(train_x, train_y)

#train预测
train_predict_y = SVM.predict(train_x)
train_predict_proba = SVM.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(SVM, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = SVM.predict(test_x)
test_predict_proba = SVM.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(SVM, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0],test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", SVM.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)

plt.sca(ROC)
ROC.plot(fpr,
         tpr,
         lw=lw,
         label='SVM ROC curve (AUC=%0.3f)' % test_auc
         )

plt.sca(PR)
plt.plot(recalls, precisions, label='SVM PR curve')
print("==============================================================================================================================")

print("XGBIMP============================================================================================================================")

XGBIMP = XGBC(
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

XGBIMP.fit(train_x, train_y)

#train预测
train_predict_y = XGBIMP.predict(train_x)
train_predict_proba = XGBIMP.predict_proba(train_x)[:, 1]
train_acc, train_precision, train_recall, train_f1 = evaluate_a_p_r_f(train_y, train_predict_y)
train_auc = evaluate_auc(train_y, train_predict_proba)
# train_auc95ci = evaluate_auc95ci(XGBIMP, train_x, train_y, train_x, train_y, nsamples=200)
print("train_acc:{: <10.3f}".format(train_acc),
      "train_precision:{: <10.3f}".format(train_precision),
      "train_recall:{: <10.3f}".format(train_recall),
      "train_f1:{: <10.3f}".format(train_f1),
      "train_auc:{: <10.3f}".format(train_auc),
      # "train_auc95ci:", train_auc95ci
      )

#test预测
test_predict_y = XGBIMP.predict(test_x)
test_predict_proba = XGBIMP.predict_proba(test_x)[:, 1]
test_acc, test_precision, test_recall, test_f1 = evaluate_a_p_r_f(test_y, test_predict_y)
test_auc = evaluate_auc(test_y, test_predict_proba)

test_auc95ci = evaluate_auc95ci(XGBIMP, train_x, train_y, test_x, test_y, nsamples=200)
print("test_acc:{: <10.3f}".format(test_acc),
      "test_precision:{: <10.3f}".format(test_precision),
      "test_recall:{: <10.3f}".format(test_recall),
      "test_f1:{: <10.3f}".format(test_f1),
      "test_auc:{: <10.3f}".format(test_auc),
      "test_auc95ci:{:.3f}-{:.3f}".format(test_auc95ci[0],test_auc95ci[1])
      )

print("预测值",[int(i) for i in test_predict_y])
print("真值",[int(i) for i in test_y])

sum = 0
for i in range(len(test_y)):
    if (test_y[i] == test_predict_y[i]):
        sum += 1
print("手算acc:", sum / len(test_predict_y))
print("sklearn算acc:", accuracy_score(test_y, test_predict_y))
print("模型算acc:", XGBIMP.score(test_x, test_y))

# 混淆矩阵
p1_a1 = 0
p1_a0 = 0
p0_a1 = 0
p0_a0 = 0

for i in range(len(test_predict_y)):
    if (test_predict_y[i] == 0 and test_y[i] == 0):
        p0_a0 += 1
    elif (test_predict_y[i] == 0 and test_y[i] == 1):
        p0_a1 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 0):
        p1_a0 += 1
    elif (test_predict_y[i] == 1 and test_y[i] == 1):
        p1_a1 += 1

print("混淆矩阵:")
print(p0_a0, p0_a1)
print(p1_a0, p1_a1)

precisions, recalls, thresholds = precision_recall_curve(test_y, test_predict_proba)
# for i in range(len(precisions)):
#     print(precisions[i], recalls[i], thresholds[i])

for i in range(1, len(precisions)):
    print(precisions[i],recalls[i],thresholds[i-1])


fpr, tpr, thresholds = roc_curve(test_y, test_predict_proba)


plt.sca(ROC)
ROC.plot(fpr,
         tpr,
         lw=lw,
         label='ImprovedXGB ROC curve (AUC=%0.3f)' % test_auc
         )

plt.sca(PR)
plt.plot(recalls, precisions, label='ImprovedXGB PR curve')

features = list(test_x.columns)
print(XGBIMP.feature_importances_)
print(features)
print("==============================================================================================================================")

#特征重要性
plt.sca(PR)
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
