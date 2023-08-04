import pandas as pd

# data_train = pd.read_csv("./训练集.csv", encoding="gbk")
# data_test = pd.read_csv("./测试集.csv", encoding="gbk")
#
# data_train = data_train.sample(frac=1)
# data_test = data_test.sample(frac=1)
#
# data_train.to_csv('../shuffle/训练集.csv', index=False)
# data_test.to_csv('../shuffle/测试集.csv', index=False)

# data_train = pd.read_csv("./smote_train.csv", encoding="gbk")
# data_test = pd.read_csv("./smote_test.csv", encoding="gbk")
#
# data_train = data_train.sample(frac=1)
# data_test = data_test.sample(frac=1)
#
# data_train.to_csv('../shuffle/smote_train.csv', index=False)
# data_test.to_csv('../shuffle/smote_test.csv', index=False)

data_train = pd.read_csv("./low_train.csv", encoding="gbk")
data_test = pd.read_csv("./low_test.csv", encoding="gbk")

data_train = data_train.sample(frac=1)
data_test = data_test.sample(frac=1)

data_train.to_csv('../shuffle/low_train.csv', index=False)
data_test.to_csv('../shuffle/low_test.csv', index=False)