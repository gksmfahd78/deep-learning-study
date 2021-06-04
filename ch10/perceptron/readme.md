# 01) 퍼셉트론
## 1. 퍼셉트론
퍼셉트론은 프랑크 로젠 트블라트가 1957년에 제안한 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘 입니다. 퍼셉트론은 실제 뇌를 구성하는 신경 세포 뉴런의 동작과 유사한데 뉴런 가지 돌기에 신호를 받아들이고, 이 신호가 일정치 이상의 크기를 가지면 축삭돌기를 통해서 신호를 전달합니다.<br/>
![뉴런](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/%EB%89%B4%EB%9F%B0.png?raw=true)<br/>
신경 세포 뉴런의 입력 신호와 출력 신호가 퍼셉트론에서 각각 입력값과 출력값에 해당됩니다.<br>
![perceptron_final](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/perceptrin1_final.png)<br>
x는 입력값을 의미하며, W는 가중치(Weight), y는 출력값입니다. 그림 안의 원은 인공 뉴런에 해당합니다. 실제 뉴런에서의 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신합니다. 각각의 인공뉴런에서 보내진 입력값 x는 각각의 가중치 W와 함께 종착지인 인공 뉴런에 전달되고 있습니다.<br><br>
각각의 입력값에 각각의 가중치가 존재하는데, 이때 가중치의 값이 크면 클수록 해당 입력 값이 중요하다는 것을 의미합니다.<br><br>
각 입력값이 가중치와 곱해져서 인공 뉴런에 보내지고, 각 입력값과 그에 해당되는 가중치의 곱의 전체 합이 임계치(threshold)를 넘으면 종착지이에 있는 인공 뉴런을 출력신호로서 1을 출력하고 그렇지 않을 경우에는 0을 출력합니다. 이러한 합수를 계단 함수(step function)라고 하며, 아래 그래프는 계단 함수의 하나의 예를 보여줍니다.<br>
![step_function](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/step_function.png)<br>
이대 계단 함수에 사용된 이 임계치값을 수식으로 표현할 때는 보통 세타로 표현합니다.<br>
![임계치값](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/%EC%9E%84%EA%B3%84%EC%B9%98%EA%B0%92.JPG)<br>
단 위의 식에서 임계치를 좌변으로 넘기고 편향 b(bias)로 표현할 수도 있습니다. 편향 b 또한 퍼셉트론의 입력으로 사용됩니다. 보통 그림으로 표현할 때는 입력값이 1로 고정되고 편향b가 곱해지는 변수로 표현합니다.<br>
![perceoptron2_final](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/perceptron2_final.png)<br>
![임계치값2](https://github.com/gksmfahd78/deep-learning-study/blob/master/ch10/perceptron/img/%EC%9E%84%EA%B3%84%EC%B9%98%EA%B0%922.JPG)<br>
뉴런에서 출력값을 변경시키는 함수를 활성화 함수(Activation Function)라고 합니다. 초기 인공 신경망 모델인 퍼셉트론은 활성화 함수로 계산 함수를 사용하였지만, 그 뒤에 등장한 여러가지 발전된 신경망들은 계단 함수 외에도 여러 다양한 활성화 함수를 사용하기 시작했습니다. 시그모이드 함수, 소프트맥수 함수 또한 활성화 함수 중 하나입니다. 


