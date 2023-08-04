from sklearn import svm
import matplotlib.pyplot as plt
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
train_x, train_y, test_x, test_y = read_file(train_path = "../训练集.csv", test_path = "../测试集.csv")

# {'C': 1.5, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'}
# test_acc:0.859      test_precision:0.948      test_recall:0.760      test_f1:0.844      test_auc:0.945

#调参
param_grid = {"C":[0.5,1.0,1.5]
              ,"kernel":["linear"]
              ,"degree":[1,3,5,7]
              ,"gamma":["scale","auto"]
              }

best_param = 0
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_auc = 0

for param in tqdm(list(ParameterGrid(param_grid))):
    clf = svm.SVC(kernel=param["kernel"]
                  , C=param["C"]
                  , degree=param["degree"]
                  , gamma=param["gamma"]
                  , probability=True

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


