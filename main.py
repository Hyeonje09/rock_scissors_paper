import tensorflow.keras
import numpy as np
import cv2

#모델 로드하기
model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0)

classes = ['Scissors', 'Rock', 'Paper']

#이미지 전처리
while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.flip(img, 1) #좌우반전
    
    h, w, c = img.shape #높이, 너비, 채널

    img = img[:, 200:200+h] #학습된 이미지 크기에 캠 크기 맞추기
    img_input =  cv2.resize(img, (224, 224)) #이미지를 224, 224 사이즈로 변형

    #BGR -> RGB (teachable machine은 RGB를 사용, opencv는 BGR을 사용하기 때문에 바꿔줘야 함)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    #0-255 픽셀 값을 -1~1 사이 값으로 변환
    img_input = (img_input.astype(np.float32) / 127.0) - 1.0
    #0번 축을 추가(1, 224, 224, 3)
    img_input = np.expand_dims(img_input, axis=0)

    #모델이 카메라를 보고 인식한 값을 저장
    prediction = model.predict(img_input)
    print(prediction)

    #prediction이 저장한 세 개의 값 중 가장 확률이 높은 값을 선택
    idx = np.argmax(prediction)

    #인공지능이 예측한 값을 화면에 출력
    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
 
    cv2.imshow('result', img)

    # q 버튼을 누를 경우 캠 중단
    if cv2.waitKey(1) == ord('q'):
        break