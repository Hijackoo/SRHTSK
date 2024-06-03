"""
This is the code of SR-HTSK.
The required packages are:
 numpy, torch, sklearn, matplotlib, itertools, and pytsk.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import itertools

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK
import switchable_norm as sn

# Define random seed
torch.manual_seed(1447)
np.random.seed(1447)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm: The values of the computed confusion matrix
    - classes: The classes corresponding to each row and column in the confusion matrix
    - normalize: True to show percentages, False to show counts
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.figure(dpi=150)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2%' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Prepare dataset
x_train=np.loadtxt('xxxx.csv',dtype=np.float32,delimiter=',') ##Training data
y_train=np.loadtxt('xxxx.csv',dtype=np.int8,delimiter=',')    ##Training label
x_test=np.loadtxt('xxxx.csv',dtype=np.float32,delimiter=',') ##Testing data
y_test=np.loadtxt('xxxx.csv',dtype=np.int8,delimiter=',')  ##Testing label

n_class = len(np.unique(y_train))  # Num. of class

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Define TSK model parameters
n_rule = 20  # Num. of rules
lr = 0.001  # learning rate
weight_decay = 1e-8
consbn = True
order = 1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)

gmf = nn.Sequential(
        AntecedentGMF(in_dim=x_train.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        sn.SwitchNorm1d(n_rule),
        nn.PReLU()
    )# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = TSK(in_dim=x_train.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
    else:
        other_param.append(p)
optimizer = AdamW([
    {'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay}
], lr=lr)

# ----------------- split 10% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
# ----------------- define the earlystopping callback -----------------
EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=180, save_path="xxxx.pkl")

wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
              epochs=10, callbacks=[EACC], ur=0, ur_tau=1/n_class)
wrapper.fit(x_train, y_train)
wrapper.load("xxxx.pkl")

y_pred = wrapper.predict(x_test).argmax(axis=1)
print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))


cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0,1,2,3])


# 打印混淆矩阵
print("Confusion Matrix: ")
print(cm)

attack_types = ['Inflating','Shaking','Stable','Slipping']
plot_confusion_matrix(cm, classes=attack_types, normalize=True, title='Confusion matrix')