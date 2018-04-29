# -*-coding: utf-8-*-

#kNN algorithm : 기존에 분류되어있던 값으로 학습. 그뒤에 분류가 되어있지않은 값이 들어오면 기존의 것과 비교,
# 가장 유사한 데이터 항목표시를 보고, 그 항목 중 상위 k개(단, k<20)의 가장 유사한 데이터를 살펴보게 된다.
# 마지막으로 k개의 가장 유사한 데이터들중 다수결을 통해서 새로운 데이터의 분류항목을 결정.


'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir


# Python3 버전
def classify0(inX, dataSet, labels, k):
    #inX : 분류를 위한 벡터 / dataSet: 훈련을 위한 예제의 전체 행렬
    #labels : 분류 항목을 표시한 벡터/ k:투표에서 사용하게 될 최근접 이웃의 수.

   #유클리드거리를 이용한 거리 계산 : 다차원 공간에서 두 점간의 거리를 구하는 법
    diffMat = inX - dataSet  # 분류하고자 하는 점 inX를 데이터 집합의 행 개수만큼 복제한 후 데이터 집합의 점과의 차를 구함
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    #오름차순으로 거리정렬
    sortedDistIndices = distances.argsort()
    classCount = {}

    #가장 짧은 k거리를 투표
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #아이템 정렬
    #sorted : key인자에 정렬에 사용될 함수를 받음
    sortedClassCount = sorted(classCount, key=classCount.get, reverse=True)
    return sortedClassCount[0]


def createDataSet(): #우리의 편의를 위한 함수, 데이터 집합과 분류 항목을 생성한다.
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 수정본
#   - 파일을 한 번만 연다
#   - 즉, open(), readlines()를 한 번만 호출
def file2matrix(filename): #텍스트 파일의 데이터를 분류기가 사용할 수 있는 형태로 변환
    fr = open(filename)
    index = 0  # 반환할 행렬의 로우 인덱스
    classLabelVec = []  # 클래스 레이블 벡터 변수 준비
    for line in fr.readlines():
        # 문자열 양 옆 공백 제거 후 탭('\t')을 기준으로 단어로 분리한 리스트
        lineList = line.strip().split('\t')

        # lineList의 마지막 컬럼값(레이블)을 클래스 레이블 벡터 변수에 추가
        classLabelVec.append(lineList[-1])

        # 파이썬에서 리스트 이해?
        feature = [float(value) for value in lineList[0:3]]

        # 처음 읽은 라인이면 returnMat 변수 초기화, 아니면 returnMat 변수에 행 추가
        # 아래 if else 문을 한 문장으로 표현
        returnMat = vstack((returnMat, feature)) \
            if index != 0 else array(feature)
        # if (index == 0):
        #    returnMat = np.array(feature)
        # else:
        #    returnMat = np.vstack((returnMat, feature))
        index += 1
    return returnMat, classLabelVec


def autoNorm(dataSet):
    # 각 특징별 최솟값, 최댓값, 범위를 구함
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    # 정규화된 배열을 구함(정규화 공식)
    normDataSet = (dataSet - minVals) / ranges

    # ranges와 minVals은 테스트 값에 대해 정규화할 필요가 있으므로 반환해야 함
    return normDataSet, ranges, minVals


"""
file: 테스트 데이터 파일 이름
ratio: 테스트 데이터 비율
k: kNN의 k 값

입력 데이터 중 [0, 총데이터*테스트비율) 범위의 데이터가 테스트 데이터로 사용된다.
"""


def datingClassTest(file, ratio=0.1, k=3):
    # 각종 변수 준비
    datingDataMat, datingLabels = file2matrix(file)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    numTotal = normMat.shape[0]  # 데이터 전체 행의 개수 (shape[1]은 전체 열의 개수)
    numTestVecs = int(numTotal * ratio)  # 테스트 데이터 개수
    errorCount = 0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:numTotal, :],
                                     datingLabels[numTestVecs:numTotal], k)
        # 교재에서는 %s 대신 %d, 교재대로 하려면 datingTestSet2.txt 파일을 사용
        print("the classifier came back with: %s, the real answer is: %s"
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
            print("!!!NOT MATCHED!!!")
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(): # 새로 테스트한 데이터가 0, 1, 2번 인지 판별
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 트레이닝 세트를 가져온다.
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

