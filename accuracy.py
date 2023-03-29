## **Training Loop**

#Helper function for calculating accuracy

import numpy as np

def flat_accuracy(preds, labels):
    pred_flat= np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(labels_flat==pred_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):

    #Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
