import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import itertools


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        return torch.eq(pred, labels_resize).sum().float().item()/labels.size(0)


class AucCal():
    def __init__(self, make_graph=False):
        self.color = ["red", "blue", "yellow"]
        self.classss = ["A", "B", "C"]
        self.make_graph = make_graph

    def cal_auc(self, pre, true, name=None):
        fpr = []
        tpr = []
        roc_auc = []
        for i in range(len(self.classss)):
            a, b, c = self.auc(pre[:, i], true[:, i])
            fpr.append(a)
            tpr.append(b)
            roc_auc.append(c)

        STR = ""
        for i in range(len(self.classss)):
            STR = STR + self.classss[i] + ":" + str(round(roc_auc[i], 3)) + " "
        print(STR)

        if self.make_graph:
            plt.figure(figsize=(10, 10), facecolor='#FFFFFF')
            plt.title("test ROC", fontsize='20')
            for i in range(len(self.classss)):
                plt.plot(fpr[i], tpr[i], label=self.classss[i]+"   auc="+str("%.4f"%roc_auc[i]), c=self.color[i], linestyle="solid", linewidth=3)
            plt.legend(loc=4, frameon=False, fontsize='16')
            plt.xlim([0, 1])
            plt.ylim([0, 1.1])
            plt.ylabel("true positive rate", fontsize='20')
            plt.xlabel("false positive rate", fontsize='20')
            plt.savefig("results/auc_" + name + ".png")
            plt.show()

        return round(np.nanmean(roc_auc), 4)

    def auc(self, pre, true):
        fpr, tpr, threshold = metrics.roc_curve(true, pre)
        roc_auc = metrics.auc(fpr, tpr)
        return fpr, tpr, roc_auc


def matrixs(pre, true, model_name):
    matrix = np.zeros((4, 4), dtype="float") #归一化图里X,Y有几个label
    for i in range(len(pre)):
        matrix[int(true[i])][int(pre[i])] += 1
    print(matrix)
    print(round(np.sum(np.diagonal(matrix))/np.sum(matrix), 4))
    MakeMatrix(matrix=matrix, name=model_name).draw()


class MakeMatrix():
    def __init__(self, matrix, name):
        self.matrix = matrix
        self.classes = ["A", "B", "C", "D"] #归一化的label
        self.classes2 = ["A", "B", "C", "D"]
        self.name = name

    def draw(self):
        plt.figure(figsize=(10, 10), facecolor='#FFFFFF') #图片大小
        self.plot_confusion_matrix(self.matrix, self.classes, normalize=True, #归一化
                                   title=self.name)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=True,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(type(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, self.classes2, rotation=90, size=16)
        plt.yticks(tick_marks, classes, size=16)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, round(cm[i, j], 2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=20)

        plt.tight_layout()
        plt.ylabel('True', size="18")
        plt.xlabel('Predict', size="18")
        plt.savefig("results/matrix_" + self.name + ".png")
        plt.show()