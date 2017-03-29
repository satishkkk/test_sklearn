# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset from webpage
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load dataset from local system
url="C:\Users\satish\Desktop\EyeDataSet.csv"
names = ['Burning', 'WateryEyes', 'ChangeInShape', 'RednessExceedThreeHour', 'BlurView', 'Redness', 'Vasculization', 'Disese']
dataset = pandas.read_csv(url, names=names)

#shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
#plt.show()

# histograms
#dataset.hist()
#plt.show()

# scatter plot matrix
#scatter_matrix(dataset)
#plt.show()


# Split-out validation dataset
array = dataset.values
X = array[:,0:7]
Y = array[:,7]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



    # Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
#print ('this is prediction')
predictions = knn.predict(X_validation)
#print (predictions)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

lda=LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
#response_string="1,0,0,1,1,0.6,0.2"
#predict = [int(e) if e.isdigit() else float(e) for e in response_string.split(',')]
#print (predict)
#print(knn.predict([predict]))



def predict_disease( data ):
	predict1 = [int(e) if e.isdigit() else float(e) for e in data.split(',')]
	return knn.predict([predict1])