import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

random_seed = 7361
random.seed(random_seed)

def read_file(train_path = "训练集.csv", test_path = "测试集.csv" ,encoding="gbk"):
    """
    读取数据集
    :param train_path:
    :param test_path:
    :return:
    """
    #数据处理
    #训练集
    data_train = pd.read_csv(train_path, encoding=encoding)
    data_train = data_train[["NAFLD","age","VAI","hdl_c","abdominal","BMI","alt_ast","alt","Gtg"]]

    #测试集
    data_test = pd.read_csv(test_path, encoding=encoding)
    data_test = data_test[["NAFLD","age","VAI","hdl_c","abdominal","BMI","alt_ast","alt","Gtg"]]

    train_x = data_train.iloc[:, 1:]
    train_y = data_train.NAFLD

    test_x = data_test.iloc[:, 1:]
    test_y = data_test.NAFLD

    return train_x,train_y,test_x,test_y

def read_smote_file(train_path = "smote_train.csv", test_path = "smote_test.csv"):
    #数据处理
    train_x, train_y, test_x, test_y = read_file(train_path = train_path, test_path = test_path)

    # #生成数据 均衡标签
    # overstamp = SMOTE(random_state=random_seed)
    #
    # train_x, train_y = overstamp.fit_resample(train_x, train_y)
    # test_x, test_y = overstamp.fit_resample(test_x, test_y)

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y

def read_low_file(train_path = "low_train.csv", test_path = "low_test.csv"):
    #数据处理
    train_x, train_y, test_x, test_y = read_file(train_path = train_path, test_path = test_path)

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y

def min_max_data(train_x, train_y, test_x, test_y):
    """

    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :return:
    """
    all_train = pd.concat([train_x, test_x])

    min_max_scaler = preprocessing.MinMaxScaler()
    all_train = min_max_scaler.fit_transform(all_train)

    all_train = pd.DataFrame(all_train)

    train_x = all_train.iloc[:len(train_y), :]
    test_x = all_train.iloc[len(train_y):, :]

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    return train_x, train_y, test_x, test_y

def evaluate_a_p_r_f(Y, predict_y):
    """
    calculate acc p r f
    :param Y:
    :param predict_y:
    :return: # acc p r f
    """
    acc = accuracy_score(Y, predict_y)
    precision = precision_score(Y, predict_y, average='binary')
    recall = recall_score(Y, predict_y, average='binary')
    f1 = f1_score(Y, predict_y, average='binary')
    return acc, precision, recall, f1

def evaluate_auc(Y, predict_proba):
    """
    caluculate auc auc%95CI
    :param Y:
    :param predict_proba:
    :return:
    """
    auc_score = roc_auc_score(Y, predict_proba)
    return auc_score

def evaluate_auc95ci(clf, X_train, y_train, X_test, y_test, nsamples=500):
    """
    evaluate_auc95ci
    :param clf:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param nsamples:
    :return:
    """
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train.iloc[idx], y_train.iloc[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    result = np.percentile(auc_values, (2.5, 97.5))
    return result

def show_roc(Y, predict_proba):
    fpr, tpr, thresholds = roc_curve(Y, predict_proba)
    auc = roc_auc_score(Y, predict_proba)

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

# roc
if __name__ == '__main__':
    # train_x, train_y, test_x, test_y = read_file()

    # print(train_x)
    # print(train_y)
    # print(test_x)
    # print(test_y)
    # train_x, train_y, test_x, test_y = read_file()
    # train_y: 0:249, 1:329
    # test_y: 0:35, 1:96

    #保存smote数据
    # #生成数据 均衡标签
    # overstamp = SMOTE(random_state=random_seed)
    #
    # train_x, train_y = overstamp.fit_resample(train_x, train_y)
    # test_x, test_y = overstamp.fit_resample(test_x, test_y)
    #
    # print(type(train_x))
    #
    # train_smote = pd.concat([train_x, train_y], axis=1)
    # test_smote = pd.concat([test_x, test_y], axis=1)
    #
    # train_smote.to_csv('./smote_train.csv', index=False)
    # test_smote.to_csv('./smote_test.csv', index=False)

    # low化数据
    train_x, train_y, test_x, test_y = read_file()

    train_data = pd.concat([train_x, train_y], axis=1)
    test_data = pd.concat([test_x, test_y], axis=1)

    # 训练集
    train_low_0 = pd.DataFrame(columns=['age', 'VAI', 'hdl_c', 'abdominal', 'BMI', 'alt_ast', 'alt', 'Gtg', 'NAFLD'])
    train_low_1 = pd.DataFrame(columns=['age', 'VAI', 'hdl_c', 'abdominal', 'BMI', 'alt_ast', 'alt', 'Gtg', 'NAFLD'])

    for i in range(len(train_data)):
        if(train_data.iloc[i, -1] == 0):
            train_low_0 = train_low_0.append(train_data.iloc[i, :], ignore_index=True)
        elif(train_data.iloc[i, -1] == 1):
            train_low_1 = train_low_1.append(train_data.iloc[i, :], ignore_index=True)

    train_low_1 = train_low_1.sample(frac=1)
    train_low_1 = train_low_1.iloc[:len(train_low_0), :]

    train_data = pd.concat([train_low_0, train_low_1], axis=0, ignore_index=True)

    # low化数据
    test_low_0 = pd.DataFrame(columns=['age', 'VAI', 'hdl_c', 'abdominal', 'BMI', 'alt_ast', 'alt', 'Gtg', 'NAFLD'])
    test_low_1 = pd.DataFrame(columns=['age', 'VAI', 'hdl_c', 'abdominal', 'BMI', 'alt_ast', 'alt', 'Gtg', 'NAFLD'])

    for i in range(len(test_data)):
        if (test_data.iloc[i, -1] == 0):
            test_low_0 = test_low_0.append(test_data.iloc[i, :], ignore_index=True)
        elif (test_data.iloc[i, -1] == 1):
            test_low_1 = test_low_1.append(test_data.iloc[i, :], ignore_index=True)

    test_low_1 = test_low_1.sample(frac=1)
    test_low_1 = test_low_1.iloc[:len(test_low_0), :]

    test_data = pd.concat([test_low_0, test_low_1], axis=0, ignore_index=True)

    #存储
    train_data.to_csv('./low_train.csv', index=False)
    test_data.to_csv('./low_test.csv', index=False)

