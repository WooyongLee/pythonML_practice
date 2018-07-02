# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 21:42:37 2018

@author: wooyo
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#유방암 데이터셋 이용
cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
print("\n유방암 데이터의 형태: {}".format(cancer.data.shape))

print("클래스별 샘플 개수:\n{}".format({
        n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state = 1)

print(X_train.shape)
print(X_test.shape)
#c총 데이터포인트 426+143개, 30개의 측정값
# 샘플 426개의 훈련세트, 143개의 테스트 세트

#전처리가 구현된 파이썬 클래스 임포트
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
#fit 메서드에 훈련 데이터 적용

scaler.fit(X_train)


#forge 데이터셋 이용(인위적으로 만든 이진 분류 데이터셋)
#X, y = mglearn.datasets.make_forge()
#산점도 그리기
#mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#plt.legend(["클래스 0", "클래스 1"], loc=4)
#plt.xlabel("첫 번째 특성")
#plt.ylabel("두 번째 특성")
#print("X.shape: {}."format(X.shape))