# -*-coding: utf-8-*-


'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 가능한 모든 분류 항목에 대한 딕셔너리 생성.
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts: # key는 마지막 칼럼이 있는 값을 가리킴.
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 밑수가 2인 로그
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    splitDataSet(dataset, axis, value)

    dataset : 분할하고자 하는 데이터 집합
    axis : 특징의 인덱스
    value : 특징의 값.
    """
    retDataSet = [] # 분할 리스트 생성.
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] # 분할한 속성을 잘라낸다.
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    * 가정
    1. 데이터가 다중(중첩) 리스트
    2. 데이터의 마지막 컬럼, 마지막 아이템이 클래스 라벨
    """
    numFeatures = len(dataSet[0]) - 1  # 마지막 칼럼은 분류 항목 표시(labels)를 위하여 쓰인다.
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # 모든 특징에 대해 반복.
        featList = [example[i] for example in dataSet]  # 이 특징의 모든 예시에 대해 리스트 생성
        uniqueVals = set(featList)  # unique values set 얻기.
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 정보이득의 계산; ie reduction in entropy
        if (infoGain > bestInfoGain):  # 지금까지의 best 정보이득과 현재 정보이득과의 비교
            bestInfoGain = infoGain  # 만약 지금것이 best라면 이것을 best 정보이득으로 세팅.
            bestFeature = i
    return bestFeature  # integer 리턴.


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 모든 분류 항목이 같을 때 멈춤
    if len(dataSet[0]) == 1:  # 데이터 셋에 속성이 더이상 없다면 멈춤
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 모든 레이블을 복사하기에, 트리에 기존에 있던 레이블들은 섞이지않는다.
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # 색인을 위한 분류 항목 표시 문자열 변환
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
