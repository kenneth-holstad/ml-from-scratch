# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from classic_algorithms import KNN

from sklearn.datasets import load_iris
iris = load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['species'] = iris.target

'''      add analysis, viz here - determine k value, etc       '''




'''                 end analysis                               '''

'''
# this also works apparently
X, y = load_iris(return_X_y = True)
'''

X = df.drop('species', axis=1).to_numpy()
y = df['species'].to_numpy()

# scaling is supposedly not important for this dataset - but could try w vs w/o

model = KNN(k=5)
model.fit(X, y)


# could optimize k-value later via k-fold cross validation, or elbow method