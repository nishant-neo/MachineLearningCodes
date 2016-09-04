
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

#training and predicting with the SVM
svm = SVC( kernel = 'linear', C = 1.0, random_state = 0)
svm.fit( X_train_std, y_train)
y_pred = svm.predict( X_test_std)
print 'Misclassified samples: %d' %  (y_test != y_pred).sum()

#accuracy metrics
print 'Accuracy: %.2f' % accuracy_score( y_test, y_pred)

#plotting decision metrics
def plot_decision_regions( X, y, classifier, test_idx = None, resolution = 0.02):
	#setup marker generator and color map
	markers = ( 's', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap( colors[:len(np.unique(y))])
	
	#plot the decision surfaces
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid( np.arange( x1_min, x1_max, resolution),
							np.arange( x2_min, x2_max, resolution))
	Z = classifier.predict( np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf( xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim( xx1.min(), xx1.max())
	plt.ylim( xx2.min(), xx2.max())
	
	#plot all samples
	X_test, y_test = X[test_idx, :], y[test_idx]
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter( x = X[y == c1, 0], y = X[y == c1, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = c1)
		 
	#hightlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, : ], y[test_idx]
		plt.scatter( X_test[:,0], X_test[:,1], c = '', alpha = 1.0, linewidth = 1, marker = 'o', s = 55, label = 'test_set' )

		
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions( X = X_combined_std, y = y_combined, classifier = svm, test_idx = range(105,150))
plt.xlabel('petal length[standardized]')
plt.xlabel('petal width[standardized]')
plt.legend( loc = 'upper left')
plt.show()




