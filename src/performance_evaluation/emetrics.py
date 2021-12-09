import subprocess

import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve


def get_aupr(Y, P):
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y > 0, 1, 0)
    Y = Y.ravel()
    P = P.ravel()
    f = open("temp.txt", 'w')
    for i in range(Y.shape[0]):
        f.write("%f %d\n" % (P[i], Y[i]))
    f.close()
    f = open("foo.txt", 'w')
    subprocess.call(["java", "-jar", "auc.jar", "temp.txt", "list"], stdout=f)
    f.close()
    f = open("foo.txt")
    lines = f.readlines()
    aucpr = float(lines[-2].split()[-1])
    f.close()
    return aucpr


def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_r2m(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))


def compute_aupr(all_targets, all_predictions, data_used):
    if data_used == "pkd":
        all_targets = [0 if n < 7 else 1 for n in all_targets]
        all_predictions = [0 if n < 7 else 1 for n in all_predictions]
    elif data_used == "kiba":
        all_targets = [0 if n < 12.1 else 1 for n in all_targets]
        all_predictions = [0 if n < 12.1 else 1 for n in all_predictions]
    else:
        raise "unknown affinity values were used"
    precision, recall, thresholds = precision_recall_curve(all_targets, all_predictions)
    aupr = metrics.auc(recall, precision)
    return aupr
