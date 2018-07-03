# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:36:07 2018

@author: wooyo
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
#릿지회귀를 사용하기 위한 임포트
#모든 특성이 출력에 주는 영향을 최소한으로 하기 위함
#이러한 제약을 규제라고 하며, 규제란 과대적합이 되지 않도록 모델을 강제로 제한함
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)

ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트  세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
