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
