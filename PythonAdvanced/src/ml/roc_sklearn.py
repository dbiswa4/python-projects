from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Plot data
def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)


if __name__ == '__main__':
    #score = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0])
    #y = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

    score = np.array([0, 0, 1, 0, 1, 1, 2, 2, 0, 0, 1])
    y = np.array([0, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1])

    generate_results(y, score);
