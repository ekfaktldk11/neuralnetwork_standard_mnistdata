#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy
import scipy.special # 시그모이드 함수(활성화 함수 값 0.00 ~ 1.00 - 0이나 1.0값은 가질수 없음) expit() 사용을 위해 scipy.special 불러오기
import matplotlib.pyplot # 행렬을 시각화하기 위한 라이브러리

# 시각화가 외부 윈도우가 아닌 현재의 노트북 내에서 보이도록 설정
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


#신경망 클래스의 정의
class neuralNetwork:
    #신경망 초기화하기
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #입력, 은닉, 출력 계층의 노드 개수 설정
        self.inodes = inputnodes # 신경망 입력 노드 수 받아오기
        self.hnodes = hiddennodes # 신경망 은닉 노드 수 받아오기
        self.onodes = outputnodes # 신경망 출력 노드 수 받아오기
        
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 가중치 초기화 코드
        # 가중치 행렬 wih와 who
        # 배열 내 가중치는 w_i_j로 표기. 노드 i에서 다음 게층의 노드 j로 연결됨을 의미 ex) w11 , w21 ,w12, w22 등
        # 첫번째 값 : 정규분포의 중심은 0.0
        # 두번째 값 : 행렬 값의 형태를 매개변수로 받음 ex) sigma, float, / pow는 표준편차를 뜻함, 
        #             표준편차는 노드로 들어오는 연결 노드의 개수에 루트를 씌우고 역수를 취한 것.(1/루트())
        # 세번째 값 : numpy 행렬
        #             가중치가 양수가 아닌 음수일 수도 있기에 범위는 0.0 ~ 1.0 범위의 숫자에 -0.5하여 
        #             실질적으로 -0.5 ~ 0.5 사이의 값을 가지도록 변경
        
        # 학습률 (learning rate)
        self.lr = learningrate
        
        # 시그모이드 함수를 활성화 함수로 사용
        self.activation_function = lambda x: scipy.special.expit(x)
        # 여기서 lambda 함수는 x를 매개변수로 전달받아 시그모이드 함수인 scipy.special.expit(x)를 반환하는 역할
        # lambda(람다로고도 함)에 의해 생성되는 함수는 이름이 없기 때문에 익명함수(anonymous function)이라고도 함
        
        pass
    
    # 신경망 학습시키기
    def train(self, inputs_list, targets_list) :
        # 입력 리스트를 2차원의 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T
        # 여기서 T는 numpy 2차원 배열의 전치(Transpose)연산을 의미 한다. 열벡터 성분을 행벡터로 바꿔줌
        # 즉, ndmin = 2로 2차원의 배열로 만들고, .T로 2차원 배열의 input_list를 행과 열을 전치시킴
        # 전치시켜주는 이유는 앞으로의 행렬곱에 있어서 각 성분에 맞게 행렬곱을 해주기 위함임
        
        # 은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 행렬의 내적(dot) 연산
        
        # 은닉 계층에서 나가는 신호를 계산 (최종적으로 시그모이드 활성화함수를 거침)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 게산
        final_outputs = self.activation_function(final_inputs)
        
        # 출력 계층의 오차는 (실제 값 - 계산 값)
        output_errors = targets - final_outputs
        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # final_outputs의 직전 항인 / final_inputs = numpy.dot(self.who, hidden_outputs)의 결과 값 행렬은 전치되어 잇으므로 결과 값의 열이 결과 값의 행임
        # 따라서 self.who.T를 사용하여 self.who를 output_errors 에 맞춰줌 (머리속이 좀 복잡해져서 이해한것이 틀릴 수도 있다!)
        
        # 은닉 계층과 출력 계층 간의 가중치 업데이트
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # 오차 역전파 함수 참고!
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        # numpy.transpose() 는 .T와 같은 역할
        pass
    
    # 신경망에 질의하기
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)
        
        
        return final_outputs


# In[19]:


# 입력, 은닉, 출력 노드의 수
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning_rate 0.1
learning_rate = 0.1

#Creating an instance of neuralNetwork
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# mnist 학습 데이터인 csv 파일을 리스트로 불러오기 / 읽기모드
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines() # 한줄씩 읽어오기 .readlines()
training_data_file.close() # 파일 사용 후 닫아주는 습관은 기본적인 것!

# 신경망 학습 / main함수 느낌

#epoch = 주기란 학습 데이터가 학습을 위해 사용되는 횟수를 의미
epochs = 5

for e in range(epochs):
    # 학습 데이터 모음 내의 모든 레코드 탐색
    for record in training_data_list:
        # 레코드를 쉼표에 의해 분리
        all_values = record.split(',')
        # 입력 값의 범위와 값 조정
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
        # 결과 값 생성 / 실제 값인 0.99 외에는 모두 0.01 / 위의 training_data_list 개수에 따라 0, 1, 2 ... 8, 9
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0]은 이 레코드에 대한 결과 값
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[23]:


# test데이터를 통해 학습된 모델이 얼마나 잘 분류하는지 하기 위한 ~
# mnist 테스트 데이터인 csv 파일을 리스트로 불러오기
test_data_file = open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 신경망 테스트하기

#신경망의 성능의 지표가 되는 성적표를 아무 값도 가지지 않도록 초기화
scorecard=[]
# scorecard 라는 비어있는 리스트를 생성!

# 테스트 데이터 모음 내의 모든 레코드 탐색
for record in test_data_list:
    # 레코드를 쉼표에 의해 분리
    all_values = record.split(',')
    # 첫번째 값이 레이블의 참값
    correct_label = int(all_values[0])
    # 입력 값의 범위와 값 조정
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # 신경망에 질의
    outputs = n.query(inputs)
    # 가장 높은 값의 인덱스는 레이블의 인덱스와 일치
    label = numpy.argmax(outputs)
    # 정답 또는 오답을 리스트에 추가
    if(label == correct_label):
        # 정답인 경우 성적표에 1을 더함
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
# 정답의 비율인 성적을 계산해 출력
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


# In[ ]:




