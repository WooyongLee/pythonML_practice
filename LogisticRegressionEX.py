# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:14:53 2018

@author: wooyo
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train) #C = 1(디폴트)
print("C=1일 때 훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))
#기본값 C=1일 때 테스트세트와 훈련세트 성능이 비슷하므로 과소적합인 것 같음

logreg_100 = LogisticRegression(C=100).fit(X_train, y_train)
print("\nC=100일 때 훈련 세트 점수: {:.3f}".format(logreg_100.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg_100.score(X_test, y_test)))

logreg_001 =  LogisticRegression(C=0.01).fit(X_train, y_train)
print("\nC=0.01일 때 훈련 세트 점수: {:.3f}".format(logreg_001.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg_001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg_100.coef_.T, '^', label="C=100")
plt.plot(logreg_001.coef_.T, 'V', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5,5)
#plt.xlabel("특성")
#plt.ylabel("계수 크기")
plt.legend()



