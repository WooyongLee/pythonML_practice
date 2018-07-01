# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 16:51:31 2018

@author: wooyo
"""
#붓꽃 데이터셋 가져오기
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#키들 출력해보기
print("iris_dataset의 키 : \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193] + "\n...")

#예측하려는 붓꽃 품종의 이름을 문자열 배열로 갖고 있음
print("타깃의 이름 : {}".format(iris_dataset['target_names']))

print("특성의 이름 : {}".format(iris_dataset['feature_names']))

#data필드 : 꽃잎의 길이, 폭, 꽃받침의 길이와 폭을 수치값으로 갖고 있는 Numpy배열
print("data type: {}".format(type(iris_dataset['data'])))

#머신러닝에서 각 아이템은 샘플, 속성은 특성이라고 함
#즉 data배열의 크기는 샘플의 수 * 특성의 수임
print("data의 처음 다섯 행 :\n{}".format(iris_dataset['data'][:5]))

#target배열 확인
print("target의 타입: {}".format(type(iris_dataset['target'])))

# 붓꽃의 종류에 대한 데이터들
print("target의 크기 : {}".format(iris_dataset['target'].shape))
print("타깃:\n{}".format(iris_dataset['target']))

import pandas as pd
from sklearn.model_selection import train_test_split
#train_test_split 함수로 유사 난수 생성기 이용하여 데이터 무작위 섞기
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

#trainset 75%
print("X_train 크기: {}".format(X_train.shape))
print("y_train 크기: {}".format(y_train.shape))

#testset 25%
print("X_test 크기: {}".format(X_test.shape))
print("y_test 크기: {}".format(y_test.shape))

#k-Nearest Neighbors Classfier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
#매개변수 : 이웃의 갯수

#train dataset으로부터 모델을 만들기 위해 knn객체의 fit 메서드 사용
knn.fit(X_train, y_train)

#예측하기, 붓꽃데이터를 직접 넣어서
X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nX_new.shape: {}".format(X_new.shape))

#knn객체의 predict 메서드 이용하기
prediction = knn.predict(X_new)
print("\n예측 : {}".format(prediction))
print("예측한 타깃의 이름: {}". format(
        iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("테스트 세트에 대하 예측값 :\n {}".format(y_pred))
print("테스트 객체의 정확도 : {:.2f}".format(np.mean(y_pred == y_test)))

#knn 객체의 score 메서드로 테스트 세트의 정확도 계산하기
print("테스트 세트에 정확도 : {:.2f}".format(knn.score(X_test, y_test)))

