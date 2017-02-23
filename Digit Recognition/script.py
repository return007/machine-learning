import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import csv

from readcsv import getcsvdata

data = getcsvdata("train.csv")
print "Data Load Success!!!"

feature = data[:,1:785]
label = data[:,0]

feature_train, feature_test, label_train, label_test = cross_validation.train_test_split(feature, label, test_size=0.15, random_state = 33)

fig = plt.figure("Sample")

for i in range(8):
	a = fig.add_subplot(4,2,i+1)
	plt.imshow(feature_train[i].reshape((28,28)))
plt.show()

#print "Data Visualization Success!!!"
print "Press Enter to continue : "
raw_input()

# Train the classifier
clf = KNeighborsClassifier()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)

acc = accuracy_score(pred, label_test)
print("Accuracy : ",acc)

fig = plt.figure("Test Results Sample")
for i in range(8) :
	a = fig.add_subplot(2,4,i+1)
	a.set_title(str(pred[i]))
	plt.imshow(feature_test[i].reshape((28,28)))
plt.show()

data = getcsvdata("test.csv")
feature = data

pred = clf.predict(feature)
with open("Predictions.csv","w") as predictions : 
	predictions_writer = csv.writer(predictions)
	predictions_writer.writerow(['ImageId','Label'])
	
	for i in range(0,len(pred)) : 
		predictions_writer.writerow([ i+1, (int)(pred[i])])

predictions.close()

print("Predictions done!!!")