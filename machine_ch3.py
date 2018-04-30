# -*-coding: utf-8-*-

"""
* 의사결정 트리는 마치 스무고개 게임처럼 동작한다.


* Decision Tree Flowchart (그림 3.1 p.49)
  - Decision Block (의사결정 블록, 사각형)
  - Terminal Block (단말 블록, 타원형)
  - Branch (가지)


* 장점
  - 적은 계산 비용
  - 이해하기 쉬운 학습 결과
  - 누락된 값 있어도 처리 가능
  - 분류와 무관한 특징도 처리 가능


* 단점
  - 과적합(overfitting)되기 쉬움: 너무 복잡한 의사결정 트리


* 적용
  - 수치형 값, 명목형 값

의사결정 트리의 일반적인 접근
- 이런 방법으로 트리를 만드는 알고리즘은 단지 명목형 값만 처리할 수 있기 때문에,
   연속형 값은 양자화 되어야한다. (150-160, 161-170 이런식으로 범위를 주어주는 것.)

 ID3(Iterative Dichotomiser 3) 알고리즘
1. 데이터를 가장 잘 나눌 수 있는 특징을 먼저 찾아서 데이터 집합을 하위 집합으로 분할
  - 정보 이득(Information Gain)이 가장 큰 특징
  - 엔트로피(Entopy)가 가장 크게 낮아지는 특징
2. 해당 특징을 포함하는 노드 생성
3. 하위 집합의 모든 데이터가 같은 클래스에 속하면 해당 하위 집합에 대한 분류 종료
4. 2의 경우가 아니라면 이 하위 집합에 대해 1을 적용
5. 모든 데이터가 분류될 때까지(= 모든 하위 집합에 대해) 1~4 반복
  - 재귀적 방법으로 해결

가장 적합한 분할 기준을 선택하는 방법

1. 정보 이득 : 데이터를 분할하기 전과 후의 정보량(엔트로피) 변화
               * 엔트로피
                * 정보에 대한 기댓값
                * 불확실한 정도, 무질서 정도
                * 확률이 낮은 사건이 많을수록 정보의 엔트로피(불확실성)이 커진다
                * 정보의 불확실성(엔트로피)가 높다
                    - 어떤 값(정보)가 나올 지 알기 힘들다
                * 엔트로피가 높은 원인
                    - 모든 사건의 확률이 균등하다
                    - 확률이 낮은 사건이 많다
                    - 정보가 다양하다
               * 개별 정보량
                * 확률이 낮을수록 개별 정보량은 커진다 == 엔트로피가 커지는데 기여
                    - 로그의 결과에 -1을 곱한 이유
                * 밑이 2
                    - 정보를 전달(표현)하는데 몇 자리 2진수(몇 비트)면 충분한가

2. 지니 불순도
3. 분산 감소

"""

# 데이터 집합의 섀넌 엔트로피를 계산하는 함수

import ch3_trees

myDat, labels = ch3_trees.createDataSet()

# print(myDat)
# print(labels)

# print(ch3_trees.calcShannonEnt(myDat)) # 섀넌 엔트로피 계산.

myDat[0][-1] = 'maybe'

# print(ch3_trees.calcShannonEnt(myDat))

# 값이 높아진 것은 maybe가 추가됨으로, 데이터가 더더욱 불확실해졌음을 말해준다.


# splitting the dataset
"""
splitDataSet(dataset, axis, value)

dataset : 분할하고자 하는 데이터 집합
axis : 특징의 인덱스
value : 특징의 값.
"""

# print(ch3_trees.splitDataSet(myDat, 0, 1))
# print(ch3_trees.splitDataSet(myDat, 0, 0))

# 데이터를 자르는데 가장 좋은 특징을 고르자!

# print(ch3_trees.chooseBestFeatureToSplit(myDat)) # 0번이 데이터를 분류하는데 가장좋은 속성이다...

# print(myDat)

"""
Recursively building the tree

  1. 최선의 분할 특징으로 데이터 집합을 분할
  2. 이진 트리가 아니므로 2개 이상으로도 분할 가능
  3. 브랜치를 따라 하위 노드로 이동
  4. 1~3의 과정을 반복 ==> 재귀적 호출
  
  
  * 재귀적 호출에 대한 재귀 중단 조건
    - 브랜치의 모든 사례가 같은 레이블일 경우
    - 분할할 특징이 더 이상 남아있지 않은 경우
      + 레이블 중에서 다수결에 의해 결정
      + 다수결 코드 (p.61)
        
        
  * 재귀가 중단된 지점의 노드
    - 리프(leaf) 노드, 말단 블록(terminating block)
    """

myTree = ch3_trees.createTree(myDat, labels)
# print(myTree)

# 텍스트 주석을 가진 트리노드 플롯하기.

import ch3_treePlotter

# ch3_treePlotter.createPlot()


# 주석 트리 구축하기.


# print(ch3_treePlotter.retrieveTree(1))
myTree = ch3_treePlotter.retrieveTree(0)
# print(myTree)

# print(ch3_treePlotter.getNumLeafs(myTree))
# print(ch3_treePlotter.getTreeDepth(myTree))

# ch3_treePlotter.createPlot(myTree)

myTree['no surfacing'][3] = 'maybe'
# print(myTree)

# ch3_treePlotter.createPlot(myTree)

myDat, labels = ch3_trees.createDataSet()

print(labels)

myTree = ch3_treePlotter.retrieveTree(0)

print(myTree)

print(ch3_trees.classify(myTree, labels, [1, 0]))  # 이것이 물고기인가? 라는 질문에 대한 답
print(ch3_trees.classify(myTree, labels, [1, 1]))

'''
의사결정 트리 유지하기

* 분류 한 문제마다 의사결정 트리를 작성하는 것은 시간 낭비
* 훈련된 의사결정 트리를 저장해두었다가 필요할 때 불러내어 사용


pickle 모듈

* 파이썬 객체 구조를 직렬화/역직렬화하기 위한 바이너리 프로토콜 구현
  즉, 객체를 계속적으로 사용한다는 것은, 사용한 후에 저장한다는 것을 의미.
  
  결과적으로 한 번 학습시켰던 것을 다시 사용할때는 또 학습시켜줄 필요가 없게 된다는 소리이다.

* pickling: 파이썬 객체 ==> 바이트 스트림
  - serialization, marshalling, flattening
* unpickling: 바이트 스트림 ==> 파이썬 객체


* pickle.dump()
* pickle.load()
'''

ch3_trees.storeTree(myTree, 'classifierStorage.txt')
a = ch3_trees.grabTree('classifierStorage.txt')
print(a)

# 컨텍트 렌즈 유형 예측.
'''
렌즈 데이터 집합
lenses.txt 데이터 파일
age
prescript
astigmatic (난시)
tear rate
'''

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = ch3_trees.createTree(lenses, lensesLabels)

print(lensesTree)

ch3_treePlotter.createPlot(lensesTree)

print(ch3_trees.classify(lensesTree, lensesLabels, ['young', 'hyper', 'no', 'normal']))

print(ch3_trees.classify(lensesTree, lensesLabels, ['young', 'hyper', 'no', 'reduced']))
