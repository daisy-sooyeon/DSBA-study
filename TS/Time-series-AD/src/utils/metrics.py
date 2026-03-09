import numpy as np
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, precision_score, recall_score, average_precision_score
import sklearn.metrics
from collections import Counter
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from math import floor, log
import pandas as pd

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    true = np.where(true == 0, 1e-5, true)  # Avoid division by zero
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    true = np.where(true == 0, 1e-5, true)  # Avoid division by zero
    return np.mean(np.square((pred - true) / true))

def cal_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    
    metric_dict = {'MAE': mae,
                   'MSE': mse,
                   'RMSE': rmse,
                   'MAPE': mape,
                   'MSPE': mspe,}
                #    'RSE': rse}
    
    print('mse: {:.4f}\t mae: {:.4f}\t mape: {:.4f}'.format(mse, mae, mape))
    return metric_dict

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    auc = safe_auc(actual, predict)
    return f1, precision, recall, TP, TN, FP, FN, auc


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
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


def PA_percentile(score, label,
                  threshold=None,
                  pred=None,
                  K=100,
                  calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    anomalies = []

    for i in range(len(actual)):
        if actual[i]:
            if not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                anomalies.append([i, i])
            else:
                anomalies[-1][-1] = i
        else:
            anomaly_state = False

    for i, [start, end] in enumerate(anomalies):
        collect = Counter(predict[start:end + 1])[1]
        anomaly_count += collect
        collect_ratio = collect / (end - start + 1)

        if collect_ratio * 100 >= K and collect > 0:
            predict[start:end + 1] = True
            latency += (end - start + 1) - collect

    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float('nan')

def safe_auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return float('nan')
    
def calc_seq(score, label, threshold, K=0, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        roc_auc = safe_auc(label, score)
        auprc = safe_auprc(label, score)
        #predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict, latency = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        t.append(latency)
        return t
    else:
        roc_auc = safe_auc(label, score)
        auprc = safe_auprc(label, score)
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(roc_auc)
        t.append(auprc)
        return t


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-4, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=False, verbose=False)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pot-AUC': p_t[7]
    }


# here for our refined best-f1 search method
def get_best_f1(score, label):
    """
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    """

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    return (best_f1_res,
            best_TP / (best_P + 1e-5),
            best_TP / (tot_anomaly + 1e-5),
            best_TP,
            score.shape[0] - best_P - tot_anomaly + best_TP,
            best_P - best_TP,
            tot_anomaly - best_TP), threshold


# calculate evaluation metrics (best-F1, AUROC, AP) under point-adjust approach.
def get_adjusted_composite_metrics(score, label):
    score = -score  # change the recons prob to anomaly score, higher anomaly score means more anomalous
    # adjust the score for segment detection. i.e., for each ground-truth anomaly segment, use the maximum score
    # as the score of all points in that segment. This corresponds to point-adjust f1-score.
    assert len(score) == len(label)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])

    # now get the adjust score for segment evaluation.
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=label, y_score=score, drop_intermediate=False)
    auroc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true=label, probas_pred=score)
    # validate best f1
    f1 = np.max(2 * precision * recall / (precision + recall + 1e-5))
    ap = sklearn.metrics.average_precision_score(y_true=label, y_score=score, average=None)
    return auroc, ap, f1, precision, recall, fpr, tpr, threshold


def anomaly_metric(scores, true):
    fpr, tpr, thresholds = roc_curve(true, scores, pos_label=1)
    J = tpr - fpr
    ix = np.argmax(J)
    pred = np.where(scores < thresholds[ix], 0, 1)
    precision = precision_score(true, pred, pos_label=1)
    recall = recall_score(true, pred, pos_label=1)
    f1 = f1_score(true, pred, pos_label=1, average='micro')
    auroc = safe_auc(true, scores)

    return precision, recall, f1, auroc

def percentile_search(combined_energy, score, label, anomaly_ratio):
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
    target = calc_seq(score, label, threshold, calc_latency=True)
    return target, threshold


def bf_search1(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq1(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    return m, m_t

def calc_seq1(score, label, threshold, K=0, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        # roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        #predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict, latency = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        # t.append(roc_auc)
        t.append(auprc)
        t.append(latency)
        return t
    else:
        # roc_auc = roc_auc_score(label, score)
        auprc = average_precision_score(label, score)
        # predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        predict = PA_percentile(score, label, threshold, K=K, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        # t.append(roc_auc)
        t.append(auprc)
        return t




class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

        Parameters
        ----------
        q
            Detection level (risk)
    
        Returns
        ----------
        SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object
        
        Parameters
        ----------
        init_data : list, numpy.array or pandas.Series
            initial batch to calibrate the algorithm
            
        data : numpy.array
            data for the run (list, np.array or pd.series)
    
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        
        Parameters
        ----------
        data : list, numpy.array, pandas.Series
            data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step
        
        Parameters
        ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t 
        verbose : bool
            (default = True) If True, gives details about the batch initialization
        verbose: bool
            (default True) If True, prints log
        min_extrema bool
            (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
            scalar function
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            ## Bug fix - Shreshth Tuli
            if step == 0: bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
            observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
            numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q
        
        Parameters
        ----------
        gamma : float
            GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
        Run SPOT on the stream
        
        Parameters
        ----------
        with_alarm : bool
            (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'
            
            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms
            
        """
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in range(self.data.size):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run
        
        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
            (default = True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots
            
        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig
    
def pot_eval(init_score, score, label, q=1e-3, level=0.02):
    """
    init_score: Train 데이터의 Anomaly Score (정상 분포 기준)
    score: Test 데이터의 Anomaly Score
    label: Test 데이터의 정답지
    """
    s = SPOT(q)  # SPOT object 생성
    s.fit(init_score, score)  # Train 점수와 Test 점수 넣기
    s.initialize(level=level, min_extrema=False, verbose=False)  # 초기화
    ret = s.run(dynamic=False)  # POT 임계값 계산
    
    pot_th = np.mean(ret['thresholds']) # 계산된 동적 임계값 평균
    
    # 계산된 임계값으로 Point-Adjust 적용 및 F1 계산
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    
    print(f'POT result: F1 {p_t[0]:.4f}, Threshold {pot_th:.4f}, Latency {p_latency:.4f}')
    
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'threshold': pot_th,
        'latency': p_latency,
        'AUC': p_t[7]
    }, pred


def hit_att(score, label, P=1.0, dict_key=None):
    """
    Hit Rate@P%: Top P * (실제 이상치 개수) 개의 높은 점수 중, 
    실제 이상치가 얼마나 포함되어 있는지 비율을 계산합니다.
    """
    if dict_key is None:
        dict_key = f'HitRate@{P*100:.0f}%'
        
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()
    
    total_anomalies = int(np.sum(label))
    if total_anomalies == 0:
        return {dict_key: 1.0}
    
    top_k = int(total_anomalies * P)
    top_k = min(top_k, len(score))
    
    top_k_indices = np.argsort(score)[::-1][:top_k]
    
    hits = np.sum(label[top_k_indices])
    hit_rate = hits / (total_anomalies + 1e-5)
    
    return {dict_key: hit_rate}


def ndcg(score, label, P=1.0, dict_key=None):
    """
    NDCG@P%: Top P * (실제 이상치 개수) 개의 순위를 평가합니다.
    """
    if dict_key is None:
        dict_key = f'NDCG@{P*100:.0f}%'
        
    score = np.asarray(score).flatten()
    label = np.asarray(label).flatten()
    
    total_anomalies = int(np.sum(label))
    if total_anomalies == 0:
        return {dict_key: 1.0}
    
    top_k = int(total_anomalies * P)
    top_k = min(top_k, len(score))
    
    sorted_indices = np.argsort(score)[::-1]
    sorted_labels = label[sorted_indices][:top_k]
    
    ranks = np.arange(1, top_k + 1)
    dcg = np.sum(sorted_labels / np.log2(ranks + 1))
    
    ideal_hits = min(top_k, total_anomalies)
    ideal_ranks = np.arange(1, ideal_hits + 1)
    idcg = np.sum(1.0 / np.log2(ideal_ranks + 1))
    
    ndcg_val = dcg / (idcg + 1e-5)
    
    return {dict_key: ndcg_val}

def log_metrics(modelname, dataname, metrics, logger):
    logger.info(f"{modelname} model, {dataname} dataset")

    for k, v in metrics.items():
        logger.info(f"  - {k:<6}: {v:.4f}" if isinstance(v, float) else f"  - {k:<6}: {v:.4f}")