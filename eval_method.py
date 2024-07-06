#eval_method.py
import numpy as np

def calc_point2point(predict, actual):
    """
    Calculate F1 score, precision, recall, and other metrics given the predictions and actual labels.

    Args:
        predict (np.ndarray): The predicted labels.
        actual (np.ndarray): The actual labels.

    Returns:
        tuple: F1 score, precision, recall, true positives (TP), true negatives (TN), false positives (FP), false negatives (FN).
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)

    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return f1, precision, recall, TP, TN, FP, FN

def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Adjust predicted labels using the given score and threshold, or directly use provided predictions.

    Args:
        score (np.ndarray): The anomaly scores.
        label (np.ndarray): The ground-truth labels.
        threshold (float): The threshold of anomaly score. Points with scores higher than the threshold are labeled as "anomaly".
        pred (np.ndarray, optional): If provided, adjust this prediction array instead of using `score` and `threshold`.
        calc_latency (bool): Whether to calculate latency.

    Returns:
        np.ndarray: Adjusted predicted labels.
        float: Latency (if calc_latency is True).
    """
    if len(score) != len(label):
        raise ValueError("Score and label must have the same length")

    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    thresh = np.percentile(score, float(threshold))

    if pred is None:
        predict = score > thresh
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate F1 score for a sequence of scores and labels.

    Args:
        score (np.ndarray): The anomaly scores.
        label (np.ndarray): The ground-truth labels.
        threshold (float): The threshold for labeling anomalies.
        calc_latency (bool): Whether to calculate latency.

    Returns:
        list: Metrics including F1 score, precision, recall, etc.
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        metrics = list(calc_point2point(predict, label))
        metrics.append(latency)
        return metrics
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)

def bf_search(score, label, step_num):
    """
    Perform a brute-force search to find the best F1 score by adjusting the threshold.

    Args:
        score (np.ndarray): The anomaly scores.
        label (np.ndarray): The ground-truth labels.
        step_num (int): Number of steps for the search.

    Returns:
        tuple: Best F1 score and corresponding threshold.
    """
    threshold = 93
    best_metrics = (-1., -1., -1.)
    best_threshold = 0.0

    for i in range(step_num):
        current_metrics = calc_seq(score, label, threshold, calc_latency=True)
        if current_metrics[0] > best_metrics[0]:
            best_threshold = threshold
            best_metrics = current_metrics

        threshold += 0.01

    return best_metrics, best_threshold
