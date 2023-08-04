原始数据标签分布:  
    train_y: 0:249, 1:329  
    test_y: 0:35, 1:96


原始数据+调参:

LR:  
test_acc:0.824      test_precision:0.910      test_recall:0.844      test_f1:0.876      test_auc:0.917      test_auc95ci: [0.8764881  0.93691964]  
train_acc:0.775      train_precision:0.790      train_recall:0.824      train_f1:0.807      train_auc:0.839      train_auc95ci: [0.8764881  0.93691964]  
RFC:  
test_acc:0.878      test_precision:0.908      test_recall:0.927      test_f1:0.918      test_auc:0.931      test_auc95ci: [0.88674851 0.94063244]  
train_acc:0.856      train_precision:0.840      train_recall:0.924      train_f1:0.880      train_auc:0.921      train_auc95ci: [0.88674851 0.94063244]  
SVM:  
test_acc:0.832      test_precision:0.911      test_recall:0.854      test_f1:0.882      test_auc:0.921      test_auc95ci: [0.88482143 0.93951265]  
train_acc:0.768      train_precision:0.810      train_recall:0.775      train_f1:0.792      train_auc:0.845      train_auc95ci: [0.88482143 0.93951265]  
GBC:  
test_acc:0.901      test_precision:0.946      test_recall:0.917      test_f1:0.931      test_auc:0.948      test_auc95ci: [0.89373512 0.95633557]  
train_acc:0.810      train_precision:0.814      train_recall:0.863      train_f1:0.838      train_auc:0.872      train_auc95ci: [0.89373512 0.95633557]  
XGB:  
test_acc:0.916      test_precision:0.947      test_recall:0.938      test_f1:0.942      test_auc:0.945      test_auc95ci: [0.880625   0.95372768]  
train_acc:0.808      train_precision:0.810      train_recall:0.866      train_f1:0.837      train_auc:0.877      train_auc95ci: [0.880625   0.95372768] 


smote数据+调参  
RFC:  
test_acc:0.818      test_precision:0.796      test_recall:0.854      test_f1:0.824      test_auc:0.930      test_auc95ci: [0.89365234 0.94314779]  
train_acc:0.848      train_precision:0.809      train_recall:0.912      train_f1:0.857      train_auc:0.919      train_auc95ci: [0.89365234 0.94314779]  
GBC:  
test_acc:0.865      test_precision:0.865      test_recall:0.865      test_f1:0.865      test_auc:0.938      test_auc95ci: [0.89474826 0.95284424]  
train_acc:0.793      train_precision:0.814      train_recall:0.760      train_f1:0.786      train_auc:0.877      train_auc95ci: [0.89474826 0.95284424]  
XGB:  
test_acc:0.859      test_precision:0.871      test_recall:0.844      test_f1:0.857      test_auc:0.945      test_auc95ci: [0.8951199  0.95058865]  
train_acc:0.780      train_precision:0.805      train_recall:0.739      train_f1:0.770      train_auc:0.884      train_auc95ci: [0.8951199  0.95058865]  
LR:  
test_acc:0.854      test_precision:0.905      test_recall:0.792      test_f1:0.844      test_auc:0.926      test_auc95ci: [0.89430881 0.94417589]  
train_acc:0.777      train_precision:0.790      train_recall:0.754      train_f1:0.771      train_auc:0.849      train_auc95ci: [0.89430881 0.94417589]  
SVM:  
test_acc:0.839      test_precision:0.911      test_recall:0.750      test_f1:0.823      test_auc:0.921      test_auc95ci: [0.90022515 0.9442844 ]  
train_acc:0.767      train_precision:0.793      train_recall:0.723      train_f1:0.757      train_auc:0.853      train_auc95ci: [0.90022515 0.9442844 ]  



