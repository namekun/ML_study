# -*-coding: utf-8-*-

import ch2_kNN

group, labels = ch2_kNN.createDataSet()

print(group)
print(labels)

d = ch2_kNN.classify0([1, 1], group, labels, 3)

print(d)

datingDataMat, datingLabels = ch2_kNN.file2matrix('datingTestSet.txt')

print(datingDataMat[:, 2])
print(datingLabels[0:2])

# 비디오게임 x 아이스크림 소비량 으로 산점도 나타내기
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])

# plt.show()

from numpy import array
import numpy as np

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

datingDataMat2, datingLabels2 = ch2_kNN.file2matrix('datingTestSet2.txt')

# ax1.scatter(datingDataMat2[:, 1], datingDataMat2[:, 2], 15.0 * float(array(datingLabels2)), 15.0 * float(array(datingLabels2)))

datingDataMat, datingLabels = ch2_kNN.file2matrix('datingTestSet.txt')
title0 = 'The Number of Frequent Flyer Miles Earned Per Year'
title1 = 'Percentage of Time Spent Playing Video Games'
title2 = 'Liters of Ice Cream Consumed Per Week'
data0, data1, data2 = datingDataMat[:, 0], datingDataMat[:, 1], datingDataMat[:, 2]
legendString = ['Did Not Like', 'Liked in Small Doses', 'Liked in Large Doses']
output = [('b', 20) if x == 'didntLike'
          else ('g', 30) if x == 'smallDoses'
else ('r', 50)
          for x in datingLabels]

# if문이 앞뒤가 뒤바뀌어 놓여져 있다.
# (color, dot_size)

colors = [x for (x, y) in output]
markers = [y for (x, y) in output]

fig = plt.figure()

ax = fig.add_subplot(111)

# 어떤 값을 기준으로 했을때, 특징이 확연하게 드러나는가?

# x axis: Percentage of Time Spent Playing Video Games
# y axis: Liters of Ice Cream Consumed Per Week
ax.scatter(data1, data2, c=colors, s=markers, edgecolors='k')
plt.xlabel(title1)
plt.ylabel(title2)

type1 = ax.scatter([10], [-10], s=20, c='b')
type2 = ax.scatter([10], [-15], s=30, c='g')
type3 = ax.scatter([10], [-20], s=50, c='r')
ax.legend([type1, type2, type3], legendString, loc=1)
minX, maxX = min(data1), max(data1)
minY, maxY = min(data2), max(data2)
marginX, marginY = np.multiply(0.05, [maxX - minX, maxY - minY])
ax.axis([minX - marginX, maxX + marginX, minY - marginY, maxY + marginY])
# plt.show()

# x axis: The Number of Frequent Flyer Miles Earned Per Year
# y axis: Liters of Ice Cream Consumed Per Week
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data0, data2, c=colors, s=markers, edgecolors='k')
plt.xlabel(title0)
plt.ylabel(title2)
ax.legend([type1, type2, type3], legendString, loc=1)
minX, maxX = min(data0), max(data0)
minY, maxY = min(data2), max(data2)
marginX, marginY = np.multiply(0.05, [maxX - minX, maxY - minY])
ax.axis([minX - marginX, maxX + marginX, minY - marginY, maxY + marginY])
# plt.show()

# x axis: The Number of Frequent Flyer Miles Earned Per Year
# y axis: Percentage of Time Spent Playing Video Games
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data0, data1, c=colors, s=markers, edgecolors='k')
plt.xlabel(title0)
plt.ylabel(title1)
ax.legend([type1, type2, type3], legendString, loc='best')
minX, maxX = min(data0), max(data0)
minY, maxY = min(data1), max(data1)
marginX, marginY = np.multiply(0.05, [maxX - minX, maxY - minY])
ax.axis([minX - marginX, maxX + marginX, minY - marginY, maxY + marginY])
# plt.show()

dataSet, dataLabels = ch2_kNN.file2matrix('datingTestSet.txt')

# 각 특징별 최솟값, 최댓값, 범위를 구함
minVals = dataSet.min(0)  # data.min(원하는 리스트 번호) (max도 사용방법은 같다!)
maxVals = dataSet.max(0)
ranges = maxVals - minVals
#print(ranges)

# 정규화된 배열을 구함
normDataSet = (dataSet - minVals) / ranges
#print(normDataSet)

#print(ch2_kNN.datingClassTest('datingTestSet.txt'))


# 위의 것을 기반으로 하여 만든 사람 분류하는 함수.
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = ch2_kNN.file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = ch2_kNN.autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = ch2_kNN.classify0((inArr -
                                          minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:",
          classifierResult)  # for datingTestSet.txt
    #     resultList[(classifierResult) - 1])


#classifyPerson()

# 필기체 인식

testVector = ch2_kNN.img2vector('testDigits/0_13.txt')

print(testVector[0, 0:31])

# 검사 :  필기체 번호에 kNN 적용하기

from os import listdir

ch2_kNN.handwritingClassTest()