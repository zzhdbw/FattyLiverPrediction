import random
from utils import read_file, read_low_file, evaluate_a_p_r_f, evaluate_auc, evaluate_auc95ci
from sklearn.tree import DecisionTreeClassifier
random_seed = 7361
random.seed(random_seed)

# #数据处理
# train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

# #数据处理
# train_x, train_y, test_x, test_y = read_low_file(train_path = "../low_train.csv", test_path = "../low_test.csv")

# train_x = train_x.iloc[:50,:]
# train_y = train_y[:50]
# test_x = test_x.iloc[:70,:]
# test_y = test_y[:70]

# #数据处理
# train_x, train_y, test_x, test_y = read_low_file(train_path = "../low_train.csv", test_path = "../low_test.csv")


# 训练
clf = DecisionTreeClassifier(
    # criterion='gini'  # 'entropy'#信息熵  'gini'#基尼系数
    # , random_state=random_seed
    # ,splitter="random"# "best"选最好的    "random"随机选 能减缓过拟合
    # ,max_depth=10#剪枝 最大深度
    # ,min_samples_leaf=5#剪枝
    # ,min_samples_split=10#剪枝
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
