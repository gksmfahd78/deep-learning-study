# 01) 퍼셉트론

## 1. 퍼셉트론

퍼셉트론은 프랑크 로젠 트블라트가 1957년에 제안한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘 입니다. 퍼셉트론은 실제 뇌를 구성하는 신경 세포 뉴런의 동작과 유사한데 뉴런 가지 돌기에 신호를 받아들이고, 이 신호가 일정치 이상의 크기를 가지면 축삭돌기를 통해서 신호를 전달합니다.<br/><br>

![뉴런](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/%EB%89%B4%EB%9F%B0.png)<br><br>

신경 세포 뉴런의 입력 신호와 출력 신호가 퍼셉트론에서 각각 입력값과 출력값에 해당됩니다.<br><br>

![perceptron1_final](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/perceptrin1_final.png)<br><br>

x는 입력값을 의미하며, W는 가중치(Weight), y는 출력값입니다. 그림 안의 원은 인공 뉴런에 해당합니다. 실제 뉴런에서의 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신합니다. 각각의 인공뉴런에서 보내진 입력값 x는 각각의 가중치 W와 함께 종착지인 인공 뉴런에 전달되고 있습니다.<br><br><br/>

각각의 입력값에 각각의 가중치가 존재하는데, 이때 가중치의 값이 크면 클수록 해당 입력 값이 중요하다는 것을 의미합니다.<br><br><br/>

각 입력값이 가중치와 곱해져서 인공 뉴런에 보내지고, 각 입력값과 그에 해당되는 가중치의 곱의 전체 합이 임계치(threshold)를 넘으면 종착지이에 있는 인공 뉴런을 출력신호로서 1을 출력하고 그렇지 않을 경우에는 0을 출력합니다. 이러한 합수를 계단 함수(step function)라고 하며, 아래 그래프는 계단 함수의 하나의 예를 보여줍니다.<br><br>

![step_function](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/step_function.png)<br><br/>

이대 계단 함수에 사용된 이 임계치값을 수식으로 표현할 때는 보통 세타로 표현합니다.<br><br/>

![임계치값](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/%EC%9E%84%EA%B3%84%EC%B9%98%EA%B0%92.JPG)<br><br>

단 위의 식에서 임계치를 좌변으로 넘기고 편향 b(bias)로 표현할 수도 있습니다. 편향 b 또한 퍼셉트론의 입력으로 사용됩니다. 보통 그림으로 표현할 때는 입력값이 1로 고정되고 편향b가 곱해지는 변수로 표현합니다.<br><br>
![perceoptron2_final](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/perceptron2_final.png)<br><br>

![임계치값2](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/%EC%9E%84%EA%B3%84%EC%B9%98%EA%B0%922.JPG)<br><br>

뉴런에서 출력값을 변경시키는 함수를 활성화 함수(Activation Function)라고 합니다. 초기 인공 신경망 모델인 퍼셉트론은 활성화 함수로 계산 함수를 사용하였지만, 그 뒤에 등장한 여러가지 발전된 신경망들은 계단 함수 외에도 여러 다양한 활성화 함수를 사용하기 시작했습니다. 시그모이드 함수, 소프트맥수 함수 또한 활성화 함수 중 하나입니다. 

## 2. 단일 퍼셉트론
위 같은 퍼셉트론을 단일 퍼셉트론이라고 합니다. 퍼셉트론은 단일 퍼셉트론과 멀티 퍼셉트론으로 나누어지는데, 단일 퍼셉트론은 값을 보내는 단계과 보내는 단계과 값을 받아서 추력하는 두 단계로만 이루어집니다. 이때 각 단계를 보통 층이라고 부르며, 이 두 개의 층을 입력층(input layer)과 출력층(output layer)이라고 합니다.<br><br>

![perceoptron3_final](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/perceptron3_final.png)<br><br>

단일 퍼셉트론을 이용하면 AND, NAND, OR 게이트를 쉽게 구현 할 수 있습니다. 게이트 연산에 쓰이는 것은 두 개의 입력값과 하나의 출력값입니다. 예를 들어 AND 게이트의 경우에는 두 개의 입력 값이 모두 1인 경우에만 출력값이 1이 나오는 구조를 갖고 있습니다.<br><br>

![and_gate](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/andgate.png)
단일 퍼셉트론의 식을 통해 AND 게이트를 만족하는 두개의 가중치와 편향 값에는 가각 w1, w2, b라고 한다면 [0.5, 0.5, -0.7], [0.5, 0.5, -0.8] 또는 [1.0, 1.0, -1.0]등 외에도 다양한 가중치와 편향의 조합이 나올 수 있습니다.<br><br>

``` python
def AND_gate(x1, x2):
    w1=0.5
    w2=0.5
    b=-0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
```
<br><br>위의 함수에 AND 게이트의 입력값을 모두 넣어보면 오직 두 개의 입력값이 1인 경우에만 1을 출력합니다.<br><br>

``` python
AND_gate(0, 0), AND_gate(0, 1), AND_gate(1, 0), AND_gate(1, 1)
```

``` python
(0, 0, 0, 1)
```

<br><br>NAND 게이트 입니다.<br><br>

![nandgae](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/nandgate.png)<br><br>

AND 게이트를 총족하는 가중치와 편향값인 [0.5, 0.5, -0.7]에 -를 붙여서 [0.5, 0.5, 0.7]을 단일 퍼셉트론 식에 넣으면 NAND 게이트를 총족합니다.<br><br>

``` python
def NAND_gate(x1, x2):
    w1=-0.5
    w2=-0.5
    b=0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
```
``` python
NAND_gate(0, 0), NAND_gate(0, 1), NAND_gate(1, 0), NAND_gate(1, 1)
```

``` python
(1, 1, 1, 0)
```
<br><br>

NAND 게이트를 구현한 파이썬 코드에 입력값을 넣자, 두 개의 입력값이 1인 경우에만 0이 나오는 것을 확인할 수 있습니다. 퍼셉트론으로 NAND 게이트를 구현한 것입니다. [-0.5, -0.5, -0.7] 외에도 퍼셉트론이 NAND 게이트의 동작을 하도록 하는 다양한 가중치와 편향의 값들이 있을 것 입니다.<br><br>

두 개의 입력이 모두 0인 경우에 출력값이 0이고 나머지 경우에는 모두 출력값이 1인 OR 게이트 또한 적절한 가중치 값과 편향 값만 찾으면 단일 퍼셉트론의 식으로 구현할 수 있습니다.<br><br>

![orgate](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/orgate.png) <br><br>

각각 가중치와 편향에 대해서 [0.6, 0.6, -0.5]를 선택하면 OR 게이트를 충족합니다. <br><br>

``` python
def OR_gate(x1, x2):
    w1=0.6
    w2=0.6
    b=-0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1
```
``` python
OR_gate(0, 0), OR_gate(0, 1), OR_gate(1, 0), OR_gate(1, 1)
```

``` python
(0, 1, 1, 1)
```
<br><br>

이 외에도 이를 충족하는 다양한 가중치와 편향의 값이 있습니다.<br><br>

이처럼 단일 퍼셉트론은 AND 게이트, NAND 게이트, OR 게이트 또한 구현할 수 있습니다. 하지만 단일 퍼셉트론으로 구현이 불가능한 게이트가 있는데 바로 XOR 게이트입니다. XOR 게이트는 입력값 두 개가 서로 다른 값을 갖고 있을때에만 출력값이 1이 되고, 입력값 두 개가 서로 같은 값을 가지면 출력값이 0이 되는 게이트입니다. 위의 파이썬 코드에 아무리 수많은 가중치와 편향을 넣어봐도 XOR 게이트를 구현하는 것은 불가능합니다. 그 이유는 단일 퍼셉트론은 직선 하나로 두 영역을 나눌 수 있는 문제에 대해서만 구현이 가능하기 때문입니다.<br><br>

AND 게이트에 대한 단일 퍼셉트론을 시각화 한다면 <br><br>
![andgrapgate](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/andgraphgate.png)<br><br>
출력값 0을 하얀색 원, 1을 검은색 원으로 표현했습니다. AND 게이트를 충족하려면 하얀색 원과 검은색 원을 직선으로 나누게 됩니다. 마찬가지로 NAND 게이트나 OR 게이트에 대해서도 시각화를 했을 때 직선으로 나누는 것이 가능합니다.<br><br>

![orgateandnandgate](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/oragateandnandgate.png)<br><br>

XOR 게이트는 입력값 두 개가 서로 다른 값을 갖고 있을때에만 출력값이 1이 되고, 입력값 두 개가 서로 같은 값을 가지면 출력값이 0이 되는 게이트입니다. <br><br>

![xorgraphandxorgate](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/xorgraphandxorgate.png)<br><br>

하얀색 원과 검은색 원을 직선 하나로 나누는 것은 불가능합니다. 즉, eksdlf 퍼셉트론으로는 XOR 게이트를 구현하는 것이 불가능합니다. 이를 eksdlf 퍼셉트론은 선형 영역에 대해서만 분리가 가능하다고 말합니다. 다시 말하면 XOR 게이트는 직선이 아닌 곡선. 비선형 영역으로 분리하면 구현이 가능합니다.<br><br>

![xorgate_nonlinearity](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/img/xorgate_nonlinearity.png)<br><br>

위의 그림은 곡선을 사용한다면 하얀색 원과 검은색 원을 나눌 수 있음을 보여줍니다. 이제 XOR 게이트를 만들 수 있는 다층 퍼셉트론에 대해서 알아보도록 하겠습니다.

# 3. 멀티 퍼셉트론

XOR 게이트는 기존의 AND, NAND, OR 게이트를 조합하면 만들 수 있습니다. 퍼셉트론 관점에서 말하면, 층을 더 쌓으면 만들 수 있습니다. 멀티 퍼셉트론과 단일 퍼셉트론의 차이는 단일 퍼셉트론은 입력층과 출력층만 존재하지만, 멀티 퍼셉트론은 중간에 층을 더 추가하였다는 점입니다. 이렇게 입력층과 출력층 사이에 존재하는 층을 은닉층(hidden layer)이라고 합니다. 즉, 다층 퍼셉트론은 중간에 은닉층이 존재한다는 점이 단일 퍼셉트론과 다릅니다. 다층 퍼셉트론은 줄여서 MLP라고도 부릅니다.<br><br>


