# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import itertools
import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

Array = np.ndarray

import numpy as np
import pandas as pd

from ani1_interface import get_ani1data
import seaborn as sns
import matplotlib.pyplot as plt
import re

#from starter_code import rmse
def rmse(y: Array, y_pred: Optional[Array] = None) -> float:
    """Calculate the root mean squared error between y and y_pred.

    If y_pred is not provided, y is treated as the residual vector.
    """

    if y_pred is None:
        rmse = np.sqrt(np.mean(np.power(y, 2)))
    else:
        rmse = np.sqrt(np.mean(np.power((y - y_pred), 2)))

    return rmse


def get_energy(nheavy: int, nsamples ):
    raw = np.random.normal(size = (nheavy,nsamples))
    return np.sum(raw,0)

#%%
xplot = []
yplot = []
for nh in range(1,9):
    method1 = get_energy(nh, 1_000_000)
    method2 = get_energy(nh, 1_000_000)
    rms1 = rmse(method1,method2)
    xplot.append(nh)
    yplot.append(rms1)
#%%
xplot = np.array(xplot)
yplot= np.array(yplot)
plt.plot(xplot,yplot,'ro')
plt.plot(xplot,yplot/np.sqrt(xplot), 'bo')