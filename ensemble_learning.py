__author__ = 'akshaykulkarni'

from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
import numpy as np
import operator
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA

'''
Classifying Iris Flowers Using Different Classification Models
'''

np.random.seed(123)
iris            = datasets.load_iris()
X,y             = iris.data[:,1:3], iris.target

clf1            = LogisticRegression()
clf2            = RandomForestClassifier()
clf3            = GaussianNB()

for clf,label in zip([clf1,clf2,clf3],['Logistic Regression','Random Forests','Naive Bayes']):
    scores      = cross_validation.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print "Accuracy: %0.2f [%s]" %(scores.mean(),label)

print "-----------------------"
'''
Implementing the Majority Voting Rule Ensemble Classifier.

We add weights parameter, which let's us assign a specific weight to each classifier. In order to work with the weights,
we collect the predicted class probabilities for each classifier, multiply it by the classifier weight, and take the average.
Based on these weighted average probabilties, we can then assign the class label.
'''

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self,clfs,voting='hard',weights=None):

        self.clfs       = clfs
        if voting in ('hard','soft'):
            self.voting = voting
        if weights != None and len(clfs) == len(weights):
            self.weights    = weights
        else:
            self.weights    = None

        self.le         = LabelEncoder()

    def fit(self,X,y):

        for clf in self.clfs:
            clf.fit(X,y)

        self.le.fit(y)

        return self

    def predict(self,X):

        if 'soft' == self.voting:
            average     = self.predict_proba(X)
            majority    = self.le.inverse_transform(np.argmax(average,axis=1))

        else:
            self.classes    = self.predict_classes(X)
            self.classes    = np.asarray([self.classes[:,c] for c in range(self.classes.shape[1])])

            if self.weights:
                self.classes    = np.concatenate([np.tile(self.classes[:,c,None],w) for w,c in zip(self.weights,range(self.classes.shape[1]))],axis=1)

            majority        = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)),axis=1,arr=self.classes)

        return majority

    def transform(self, X):
        if self.weights:
            return self.predict_proba(X)
        else:
            return self.predict_classes(X)

    def predict_proba(self,X):

        self.probability    = np.asarray([clf.predict_proba(X) for clf in self.clfs])
        return np.average(self.probability,axis=0,weights=self.weights)

    def predict_classes(self,X):
        return np.asarray([clf.predict(X) for clf in self.clfs])


ensemble_classifier    = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')
for clf,label in zip([clf1,clf2,clf3,ensemble_classifier],['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f [%s]" % (scores.mean(),label))

'''
We will use a naive brute-force approach to find the optimal weights for each classifier to increase the prediction accuracy.
'''
df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))
i  = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):

            if 1 == len(set((w1,w2,w3))):
                continue

            ensemble_classifier     = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard',weights=[w1,w2,w3])
            scores                  = cross_validation.cross_val_score(ensemble_classifier, X, y, cv=5, scoring='accuracy')
            df.loc[i]               = [w1, w2, w3, scores.mean(),scores.std()]
            i                       = i+1

print df.sort(columns=['mean', 'std'], ascending=False)

'''
EnsembleClassifier - Pipelines
'''

class ColumnSelector(object):

    def __init__(self,cols):
        self.cols       = cols

    def transform(self,X,y=None):
        return X[:,self.cols]

    def fit(self,X,y=None):
        return self


pipeline1           = Pipeline([('sel', ColumnSelector([1])), ('clf', GaussianNB())])
pipeline2           = Pipeline([('sel', ColumnSelector([0, 1])),('dim', LDA(n_components=1)),('clf', LogisticRegression())])

ensemble_classifier = EnsembleClassifier([pipeline1,pipeline2])
scores              = cross_validation.cross_val_score(ensemble_classifier, X, y, cv=5, scoring='accuracy')
print("Accuracy: %0.2f [%s]" % (scores.mean(), label))

'''
Ensemble EnsembleClassifier
If one EnsembleClassifier is not yet enough, we can also build an ensemble classifier of ensemble classifiers.
'''

ensemble_classifier1 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[5,2,1])
ensemble_classifier2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[4,2,1])
ensemble_classifier3 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,4])

ensemble_classifier = EnsembleClassifier(clfs=[ensemble_classifier1, ensemble_classifier2, ensemble_classifier3], voting='soft', weights=[2,1,1])

scores = cross_validation.cross_val_score(ensemble_classifier, X, y, cv=5, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))