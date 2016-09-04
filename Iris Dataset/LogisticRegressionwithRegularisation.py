from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print np.unique(y)

#cross validation
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 0)

#feature scaling
sc = StandardScaler()
sc.fit( X_train) # using this the mean and the variance is calculated
X_train_std = sc.transform( X_train) # now we standardize the data using the parameteres learned from last step
X_test_std = sc.transform(X_test)

#training and predicting with the Logistic Regression
weights, params = [], []
for c in np.arange(-5,5):

	lr = LogisticRegression( C = 10**c, random_state = 0)
	lr.fit( X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:,0], label = 'petal_length')
plt.plot(params, weights[:,1], linestyle ='--', label = 'petal_width')
plt.ylabel('weights coefficient')
plt.xlabel('C')
plt.legend(loc ='upper left')
plt.xscale('log')
plt.show()






