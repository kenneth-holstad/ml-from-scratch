# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from classic_algorithms import KNN

from sklearn.datasets import load_digits
digits = load_digits()

X, y = digits.data, digits.target

# add analysis, viz here - determine k value, etc



'''                 end analysis                               '''

# scaling really shouldn't matter for this dataset - all pixels are in the same range

model = KNN(k=5)
model.fit(X, y)