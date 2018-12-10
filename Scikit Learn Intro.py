import scipy
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

iris = datasets.load_iris()
print(type(iris))
print(iris.data.shape)
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
pd.plotting.scatter_matrix(df,c=y, figsize=[8,8],
                  s=150, marker = 'D')
###### Classification ################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X, y)
####### predict ######### X_new is new data
#prediction = knn.predict(X_new)
#X_new.shape

##### Measuring Model Performance ###########

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 21,
                                                    stratify=y)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("test set predictions: \n {}".format(y_pred))
print(knn.score(X_test, y_test))
