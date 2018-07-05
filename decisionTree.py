# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 23:47:22 2018

@author: wooyo
"""

from sklearn.tree import DecisionTreeClassifier
#결정트리를 이용하기 위한 임포트
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 데이터를 단순히 트레이닝 데이터와 테스트 데이터로 분리
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train, y_train)
print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))

tree_2 = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_2.fit(X_train, y_train)
print("\n훈련 세트 정확도: {:.3f}".format(tree_2.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree_2.score(X_test, y_test)))

#트리모듈의 export_graphviz 이용하여 트리를 시각화 할 수 있음
#그래프 저장용 텍스트 파일 포맷인 .dot파일 생성
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["악성", "양성"],
               feature_names=cancer.feature_names,
               impurity=False, filled=True)

#트리를 만드는 결정에 각 틀성이 얼마나 중요한지 평가하는 특성 중요도 출력
print("\n특성 중요도:\n{}".format(tree.feature_importances_))

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
plot_feature_importances_cancer(tree)