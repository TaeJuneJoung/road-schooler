### 2019.05.29

기본 알아야할 개념

회귀 / 군집 / 은닉 마르코프 모델

오토인코더 / 강화학습 / 합성곱 신경망 / 순환 신경망 / 시퀸스2시퀸스 모델 / 유틸리티



`이미지 자동 레이블링`



머신러닝과 귀납적 학습

모델

>   파라미터

데이터(머신러닝의 일등급 객체)를 피처 벡터로 기술

>   피처벡터 : 데이터를 실제로 사용하기 위해 단순화한 것

벡터(속성)

1.  차원을 표현하는 자연수
2.  형(실수형, 정수형 등)

행렬(Matrix) : 벡터의 벡터

노드 : 벡터 내 각각의 아이템

그래프 : 오브젝터(노드)들로 이루어진 집합인데 네트워크를 나타내기 위해 엣지(edge)와 연결할 수 있음



피처 엔지니어링(Feature engineering)

: 적절한 피처의 수 설정 및 어떤 피처를 비교할지 설정하는 과정

```python
#ex) 자동차
자동차를 판별하는데 색이 필요하지 않으니,
색을 흑백으로 처리하면 성능이 크게 향상된다.

Q) 사람 판별, 비교에서 색이 필요할까?
사람에게서 색은 인종에 사용되지 않는가?
```



데이터는 많을수록 좋지만, 피처는 적절해야한다.

(너무 많은 피처 사용시, 차원의 저주(curse of dimensionality) 현상 발생)



학습:평가:검증

6:2:2



거리지표는 표준화된 접근방법으로는 유클리드 거리(Euclidian distance)를 사용



L0 노름

: 벡터에서 0이 아닌 원소 전체의 개수를 센다.

L1 노름
$$
\sum\ |x_n|\ 로\ 정의
$$
L-N 노름
$$
(\sum(x_n)^N)^{1/N}\ 로\ 정의
$$
L2 노름 이상을 사용하는 일은 거의 없음



### 학습의 종류

1.  지도학습(SL, Supervised Learning)
2.  비지도학습(UL, Unsupervised Learing)
3.  강화학습(RL, Reinforcement Learing)



#### 지도학습

: 지도관이 준비한 데이터로부터 학습하는 방법

학습을 위해서 미리 레이블링된 데이터가 필요



`g(x|θ)`

: 파라미터까지 포함해 모델을 보다 더 완성도 있게 표현한 것

"g가 주어졌을 때 x의 θ"라고 읽는다

최적의 θ를 θ*(theta star)라고 한다.
$$
θ^* = argmin_θCost(θ|X)\\
where\ Cost(θ|X) = \sum_{x\in X}||g(x|θ)-f(x)||
$$



`텐서플로 연산자`

https://www.tensorflow.org/api_guides/python/math_ops



### 2019.05.30

<https://ukayzm.github.io/face-clustering/>

/face_recognition : 캠을 이용한 사람 레이블링[학습 중]

>   ```bash
>   pip install opencv-python
>   pip install opencv-contrib-python
>   pip install cmake
>   pip install dlib
>   pip install face_recognition
>   pip install flask
>   ```
>
>   일반 컴퓨터에서는 작동하지 않아 이유 파악 중...



video_recog.py : 비디오 및 스트리밍 영상 사람 레이블링[파이썬 라이브러리]

>**사용법**
>
>```bash
> $ python video_recog.py -e "testVideo.mp4"
>```
>
>으로 실행



### 2019.06.03

`face detection`

: 컴퓨터 비전의 한 분야로 영상(image)에서 얼굴이 존재하는 위치를 알려주는 기술

다양한 크기의 얼굴을 검출하기 위해 피라미드 영상을 생성한 후, 한 픽셀씩 이동하며 특정 크기의 해당 영역이 어굴인지 아닌지를 **분류기(신경망, Neural Network), 아다부스트(Adaboost), 서포트 벡터 머신(Support Vector Machine) 등**으로 얼굴인지 아닌지를 결정



>   **컴퓨터 비전(Computer Vision)**
>
>   : 기계의 시각에 해당하는 부분을 연구하는 컴퓨터 과학의 최신 연구 분야 중 하나



초기 얼굴 검출에 사용된 특징은 **얼굴의 강도(intensity)**

인종, 조명 등에 따라 성능이 좌우됨에 무관한 특징 필요

**하르 유사 특징(Haar-like feature)** 사용

이후, **Local Binary Pattern(LBP, 국부 이진 패턴), Modified Census Transform** 등의 특징이 제안됨



`facial recognition system`

: 디지털 이미지를 통해 각 사람을 자동으로 식별하는 컴퓨터 지원 응용 프로그램

이미지에 나타나는 선택된 얼굴 특징과 안면 데이터베이스를 서로 비교함으로써 이루어짐



`ficial analysis`

: 나이, 성별 또는 감정을 결정하는 것과 같은 얼굴 특징에 대해서 사람들에 관한 것을 이해하려고 시도



`ficial tracking`

: 비디오 분석에 주로 사용되며 얼굴과 눈, 코, 입술을 프레임마다 따라감



>   **컴퓨터가 이미지를 보는 방법**
>
>   이미지의 가장 작은 요소를 픽셀 또는 그림 요소라고 함
>
>   컴퓨터는 픽셀을 색상 점으로 이해하지 못한다. 숫자로만 이해
>
>   색상을 숫자로 변환하기 위해 컴퓨터는 다양한 색상 모델을 사용



**Features**

: 특정 문제를 해결 관련된 이미지 정보의 조각

특정 작업을 이미지에 적용하면 features로 간주 될 수 있는 정보가 생성됨

이미지의 고유하거나 파생 된 속성은 작업을 해결하는 features로 사용될 수 있음







### 참고자료

<hr>

openCV

<https://opencv.org/>

웹캠 얼굴인식

<https://circlestate.tistory.com/5?category=679312>

openCV 튜토리얼 영상

<https://www.youtube.com/watch?v=PmZ29Vta7Vc>

Google AI Vision API

<https://cloud.google.com/vision/?hl=ko&tab=tab4>

OpenFace

<https://cmusatyalab.github.io/openface/>

딥러닝을 이용한 얼굴인식

<https://brunch.co.kr/@kakao-it/301>

기계학습 머신러닝

[https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-4-63ed781eee3c](https://medium.com/@jongdae.lim/기계-학습-machine-learning-은-즐겁다-part-4-63ed781eee3c)


카카오개발자들의 일지
https://www.slideshare.net/ifkakao/ss-115328045