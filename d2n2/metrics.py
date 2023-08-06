import os
import tensorflow as tf
import numpy as np




def ACC(model, x_test_sub, y_test_sub):
    _loss, _acc = model.evaluate(x_test_sub, y_test_sub, verbose=0)
    return _acc

def CWA(model, x_test_class, y_test_class):
    _loss, _acc = model.evaluate(x_test_class, y_test_class, verbose=0)
    return _acc


def UF1(model, x1, x2):
    probs1 = model(x1)
    probs2 = model(x2)
    z = probs1 - probs2
    return np.average(np.abs(z)[:, 0])    


def UF2(model, x1, x2):
    probs1 = model(x1)
    probs2 = model(x2)
    z = np.max(probs1,axis=1)-np.max(probs2,axis=1)
    return np.average(np.abs(z))


def ASR(model, x_trigger, y_trigger):
    _loss, _acc = model.evaluate(x_trigger, y_trigger, verbose=0)
    return _acc