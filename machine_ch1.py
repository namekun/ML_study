# -*-coding: utf-8-*-

from numpy import *

random.rand(4, 4) #랜덤의 4x4 행렬 생성

randMat = mat(random.rand(4,4))

print(randMat.I) #.I는 역행렬을 표현한 것.

invRandMat = randMat.I

myEve = randMat * invRandMat

print(myEve) #결과를 보면 대각선상에 있는 원소가 1. 나머지는 죄다 0

print(eye(4))  #eye() 함수는 단지 크기가 n인 단위행렬만을 생성

print(myEve - eye(4))
