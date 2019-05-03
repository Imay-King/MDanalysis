#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
seed=40
np.random.seed(seed)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix for prediction')
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


def plot_roc(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

#create atom dataset
#def create_atom_set(list,natom_value=3):
 #   return

#read file and divide dataset
first=pd.read_csv('first_tr.csv',header=None)
second=pd.read_csv('second_tr.csv',header=None)
data=pd.concat([first,second],ignore_index=True)
test=pd.read_csv('test_MD.csv',header=None)
train_data=data.iloc[:,:-2]
train_label=data.iloc[:,-1]
test_data=test.iloc[:,:-2]
a=test.iloc[:,-1]
t=test.iloc[:,-2]
y_true=np.array(a).tolist()
time_step=np.array(t).tolist()

#normalize the dataset
scalar = StandardScaler()
train_data_1=scalar.fit(train_data).transform(train_data)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(train_data_1, train_label):
# create model
    model = Sequential()
    model.add(Dense(26, input_dim=78, kernel_initializer='normal', activation='relu'))
   # model.add(Dropout(0.2))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_data_1, train_label, epochs=150, batch_size=10, verbose=0)
# evaluate the model

    scores = model.evaluate(train_data_1[test], train_label[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names, scores))
    cvscores.append(scores)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


'''
# define and fit the final model
model = Sequential()
model.add(Dense(3, input_dim=159, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(train_data_1, train_label, epochs=150, batch_size=10, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, train_data_1, train_label, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
scores = model.evaluate(train_data, train_label, verbose=0)
print(scores)
'''
#make prediction

y_pre=[]
y_probabilty=[]

for i in range(len(test_data)):
    Xnew = np.array([test_data.iloc[i,:].values])
    Xnew_1=scalar.transform(Xnew)
    ynew = model.predict_classes(Xnew_1,batch_size = 20, verbose = 1)
    y_prob=model.predict_proba(Xnew_1,batch_size = 20, verbose = 1)
    y_pre.append(ynew[0][0])
    y_probabilty.append(y_prob[0][0])
    print("X%s=%s, Predicted=%s" % (i+1,Xnew, ynew))  # the possibility to class 1
print(y_true)
#print(y_pre)
print(y_probabilty)
'''
# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = roc_curve(y_true, y_probabilty) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
plot_roc(fpr,tpr,roc_auc)

'''
plt.figure(figsize=(8,4))
plt.plot(time_step,y_true,"b--",linewidth=1,label='Lines for true labels')
plt.plot(time_step,y_pre,"r--",linewidth=1,label='Lines for predicted labels')
plt.xlabel("Timesteps")
plt.ylabel("Class")
plt.yticks([0,1])
plt.xticks(time_step)
plt.title("Line plot")
plt.show()
#Drawing confusion matrix plots
cm=confusion_matrix(y_true, y_pre)
plot_confusion_matrix(cm,target_names=['0','1'],normalize=True)
plt.show()







