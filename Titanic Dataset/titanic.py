import pandas as pd
import numpy as np
import csv as csv
import math
#from sklearn.ensemble import RandomForestClassifier 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sknn.mlp import Classifier, Layer


ids = []
def readClean( file ):

	t = pd.read_csv(file, header = 0)
	print t.dtypes
	#t['Gender'] = t['Sex'].map( lambda x: x[0].upper())
	t['Gender'] = t['Sex'].map({'female': 0, 'male' : 1}).astype(int)
	median_ages = np.zeros((2,3))

	
	for i in range(0,2):
		for j in range( 0, 3):
			median_ages[i,j] = t[(t['Gender'] == i) &  (t['Pclass'] == j+1)]['Age'].dropna().median()
	t['AgeFill'] = t['Age']
	t.head()
	for i in range(0, 2):
		for j in range(0, 3):
			t.loc[ (t.Age.isnull()) & (t.Gender == i) & (t.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
	#t['AgeFill'] = preprocessing.normalize(t['PassengerId'].values, norm='l2')
	#t['AgeIsNull'] = pd.isnull(t.Age).astype(int)
	#t['FamilySize'] = t['SibSp'] + t['Parch']
	#t['Age*Class'] = t.AgeFill * t.Pclass
	t.dtypes[t.dtypes.map(lambda x: x=='object')]
	t = t.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
	t = t.drop(['Age'], axis=1)
	# Collect the test data's PassengerIds before dropping it
	ids = t['PassengerId'].values
	#print ids
	t = t.drop(['SibSp','PassengerId'], axis = 1)
	#t = t.drop([''], axis = 1)
	print t.dtypes
	t = t.dropna()
	return t.values,ids
	
train_data,ids = readClean('train.csv')
#print train_data
test_data,ids = readClean('test.csv')

l,b = train_data.shape
mean = []
sigma =[]
mean = np.mean(train_data, axis=0)
sigma = np.std( train_data, axis=0 )
print mean[1],mean[2],mean[3],mean[4], mean[5]
print sigma[1],sigma[2],sigma[3],sigma[4], sigma[5]
print train_data.shape
print train_data
for i in range(0,l):
	for j in range(1,b):
		train_data[i,j] = abs(train_data[i,j] - mean[j]) / sigma[j]
print train_data
print train_data.dtype

l1,b1 = test_data.shape
#mean1 = []
#sigma1 =[]
#mean1 = np.mean(test_data, axis=0)
#sigma1 = np.sum( test_data, axis=0 )
#print mean1[0],mean1[1],mean1[2]
#print sigma1[0],sigma1[1],sigma1[2]
print test_data.shape
print test_data
for i in range(0,l1):
	for j in range(0,b1):
		test_data[i,j] = abs(test_data[i,j] - mean[j+1]) / sigma[j+1]		
print test_data
print test_data.dtype

print 'Training...'
# Create the random forest object which will include all the parameters for the fit
# forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees
# forest = forest.fit(train_data[0::,1::],train_data[0::,0])



		
model = LogisticRegression( penalty='l2', C = 0.07)
t = 9 * l / 10
print t, l
model = model.fit(train_data[0:t,1::],train_data[0:t,0])
cvd = model.predict(train_data[t+1::,1::]).astype(int)
count = 0
sum = 0
print cvd
for i in range(t+1,l):
	print cvd[i-t-1],
	print int(train_data[i,0])
	sum = sum + 1
	if cvd[i-t-1] == int(train_data[i,0]):
		count = count + 1
print "Accuracy ",
acc = (count * 100) / sum
print count , sum, acc
	
output = model.predict(test_data).astype(int)


#model = Classifier(
#    layers=
#        Layer("Sigmoid", units= 3),
#        Layer("Softmax")],
#    learning_rate=0.001,  regularize="L2", weight_decay = 0.0002,
#    n_iter=20)
#model.fit(train_data[0::,1::],train_data[0::,0])
#output = model.predict(test_data).astype(int) 
#oute = [ i[0] for i in output]
#print oute
#print output


# Take the same decision trees and run it on the test data
print 'Predicting...'
#output = forest.predict(test_data).astype(int)
predictions_file = open("logisticreg.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'

