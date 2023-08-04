import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import read_file, read_smote_file
random_seed = 7361
random.seed(random_seed)

#数据处理
train_x, train_y, test_x, test_y = read_file()
train_smote_x, train_smote_y, test_smote_x, test_smote_y = read_smote_file()

#降维
pca = PCA(n_components=2) #选择降维数量
pca = pca.fit(train_smote_x) #建模
train_x = pca.transform(train_x) #获得降维后数据列
train_smote_x = pca.transform(train_smote_x) #获得降维后数据列

#画布
colors=['red','blue']
fig, ax = plt.subplots()
#开始循环并画散点图
for i in range(len(train_x)):
    if(train_x[i,0] > 150):
        continue

    ax.scatter(
        train_x[i,0],
        train_x[i,1],
        c = colors[train_y[i]],
        # label=iris.target_names[i]
    )

plt.title('data')
plt.legend()
plt.show()

#画布
colors=['red','blue']
fig, ax = plt.subplots()
#开始循环并画散点图
for i in range(len(train_smote_x)):
    if(train_smote_x[i,0] > 150):
        continue

    ax.scatter(
        train_smote_x[i,0],
        train_smote_x[i,1],
        c = colors[train_smote_y[i]],
        # label=iris.target_names[i]
    )

plt.title('data')
plt.legend()
plt.show()


#画布
colors=['red','blue',"green"]
fig, ax = plt.subplots()
#开始循环并画散点图
for i in range(len(train_smote_x)):
    if(train_smote_x[i,0] > 150):
        continue

    if(i >= len(train_x)):
        ax.scatter(
            train_smote_x[i, 0],
            train_smote_x[i, 1],
            c=colors[2],
            # label=iris.target_names[i]
        )
    else:
        ax.scatter(
            train_smote_x[i, 0],
            train_smote_x[i, 1],
            c=colors[train_smote_y[i]],
            # label=iris.target_names[i]
        )
plt.title('data')
plt.legend()
plt.show()