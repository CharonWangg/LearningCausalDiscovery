import unittest
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def parallel_sort(xin, yin):
    n = len(xin)
    xin_sorted_idx = np.argsort(xin)
    yin_sorted_idx = np.argsort(yin)

    xout = xin[xin_sorted_idx]
    ysorted_by_x = yin[xin_sorted_idx]
    yout = yin

    # for ii in range(n):
    #    print("%d\t %.4f \t %.4f" %(ii, xout[ii], ysorted_by_x[ii]))

    # for fixed xin[.], further sort yin[...]
    x_prev = xout[0]
    same_x_start_idx = 0
    yout = []
    for ii in range(0, n, 1):
        x = xout[ii]
        if ((x > x_prev) or (ii == n - 1)):
            if (ii == n - 1):
                same_x_stop_idx = n - 1
            else:
                same_x_stop_idx = ii - 1

            if (same_x_start_idx == same_x_stop_idx):
                y_arr_for_same_x = ysorted_by_x[same_x_start_idx]
            else:
                y_arr_for_same_x = np.sort(ysorted_by_x[same_x_start_idx:same_x_stop_idx + 1:1])
                # print("%d, %d, %.4f" %(same_x_start_idx, same_x_stop_idx, x_prev))
                # print(ysorted_by_x[same_x_start_idx:same_x_stop_idx+1:1])
                # print(y_arr_for_same_x)

            yout = np.append(yout, y_arr_for_same_x)

            # print("%d, %d, %.4f" %(same_x_start_idx, same_x_stop_idx, x_prev))
            same_x_start_idx = ii
            x_prev = xout[ii]

    return xout, yout


def calcAUROC(xin, yin, verbose):
    xin, yin = parallel_sort(xin, yin)

    if (verbose > 0):
        for ii in range(len(xin)):
            print("%d\t %.6f \t %.6f" % (ii, xin[ii], yin[ii]))

    # Update input arrays to include extreme points (0,0) and (1,1) to the ROC plot
    xin = np.insert(xin, 0, 0)
    yin = np.insert(yin, 0, 0)
    xin = np.append(xin, 1)
    yin = np.append(yin, 1)

    n = len(xin)
    auroc = 0
    for ii in range(n - 1):
        h = xin[ii + 1] - xin[ii]
        b1 = yin[ii]
        b2 = yin[ii + 1]
        trapezoid_area = 0.5 * h * (b1 + b2)
        auroc = auroc + trapezoid_area

    return auroc, xin, yin


def calcAUPR(xin, yin):
    xin, yin = parallel_sort(xin, yin)
    ll = len(xin)

    # Update input arrays to include extreme points (0,1) and (1,0) to the precision-recall plot
    if (xin[0] > 0):
        xin = np.insert(xin, 0, 0)
        yin = np.insert(yin, 0, 1)
    if (xin[ll - 1] < 1):
        xin = np.append(xin, 1)
        yin = np.append(yin, 0)

    n = len(xin)
    aupr = 0
    for ii in range(n - 1):
        h = xin[ii + 1] - xin[ii]
        b1 = yin[ii]
        b2 = yin[ii + 1]
        trapezoid_area = 0.5 * h * (b1 + b2)
        aupr = aupr + trapezoid_area

    return aupr


def calculate_auprc(precision, recall):
    # Reverse the list as precision decreases with recall
    precision = precision[::-1]
    recall = recall[::-1]
    # Add the endpoint (0, 1) if it doesn't exist
    if recall[-1] != 0:
        recall = np.append(recall, 0)
        precision = np.append(precision, 1)
    auprc = np.sum((recall[:-1] - recall[1:]) * precision[:-1])
    return auprc


# Function to calculate AUROC and AUPRC
def calculate_auroc_auprc(fpr_list, tpr_list, recall_list, precision_list):

    auroc, fpr_list_sorted, tpr_list_sorted = calcAUROC(fpr_list, tpr_list, 0)
    auprc = calcAUPR(recall_list, precision_list)
    return auroc, auprc

# Unit test class
class TestAUCMethods(unittest.TestCase):
    def test_auroc_auprc(self):
        # Generate sample data
        y_true = np.array([0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        # Calculate AUROC and AUPRC using the function
        auroc, auprc = calculate_auroc_auprc(fpr, tpr, recall, precision)
        # Check if the calculated AUROC and AUPRC match the expected values
        self.assertEqual(auroc, roc_auc_score(y_true, y_scores))
        self.assertEqual(auprc, average_precision_score(y_true, y_scores))

if __name__ == '__main__':
    unittest.main()
