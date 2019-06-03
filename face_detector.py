import cv2, dlib, sys
import numpy as np 

# 얼굴 디텍터 모듈 초기화
detector = dlib.get_frontal_face_detector()
# 얼굴 특징점 모듈 초기화
predictor = dlib.shape_predictor('path/shape_predictor_68_face_landmarks.dat')


# load video 동영상 파일 로드
# cv2.VideoCapture(0) : 파일 이름대신 0을 넣으면 웹캠 사용 가능
cap = cv2.VideoCapture('path/girl.mp4')

# load overlay image
# cv2.imread(file, cv2.IMREAD_UNCHANGED)  : file이미지를 BGRA 타입으로 읽기 
overlay = cv2.imread('path/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# 동영상 파일 크기 조절
scaler = 0.3


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img


while True:
    # cap.read() : 동영상 파일에서 frame 단위로 읽기
    ret, img = cap.read()
    # 만약 프레임이 없으면 동영상 종료
    if not ret:
        break

    # img를 dsize 크기로 조절(resize)
    # img.shape의 0번째: , 1번째: 
    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
    # img를 ori 변수에 복사
    ori = img.copy()

    # img에서 모든 얼굴 찾기
    faces = detector(img)
    # 찾은 모든 얼굴에서 첫번째 얼굴
    face = faces[0]

    # img의 face 영역안의 얼굴 특징점 찾기
    dlib_shape = predictor(img, face)
    # dlib 객체를 numpy 객체로 변환
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # computer center and boundaries of face
    # 얼굴의 좌상단, 우하단 찾기
    # np.min() : 최소값 찾기
    top_left = np.min(shape_2d, axis=0)
    # np.max() : 최대값 찾기
    bottom_right = np.max(shape_2d, axis=0)

    # 얼굴 사이즈
    # 우하단에서 좌상단 좌표를 뺀 (x,y)길이의 가장 긴 값
    face_size = int(max(bottom_right - top_left) * 1.8)

    # 얼굴의 중심 구하기
    # .astype(np.int) : 실수형일수 있으므로 정수형으로 변환
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # 얼굴에 오버레이한 결과
    result = overlay_transparent(ori, overlay, center_x+25, center_y-25, overlay_size=(face_size, face_size))

    # visualize
    # cv2.rectangle() : 직사각형 그리기(선색, 두께, 종류)
    img = cv2.rectangle(img, pt1 = (face.left(), face.top()), pt2 = (face.right(), face.bottom()),
        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA )

    # 68개의 얼굴 특징점을 for를 이용해 점을 찍음
    for s in shape_2d:
      cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴에 좌상단, 우하단에 점 찍기
    cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # 얼굴의 중심에 점찍기
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)


    # 'img'라는 이름의 윈도우에 img를 띄우기
    cv2.imshow('img', img)
    cv2.imshow('result', result)
    # 1밀리세컨드만큼 대기 - 이걸 넣어야 동영상이 제대로 보임
    cv2.waitKey(1)
    