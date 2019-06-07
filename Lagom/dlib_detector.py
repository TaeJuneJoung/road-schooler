import dlib
# openCV
import cv2

face_detector = dlib.get_frontal_face_detector()
img = cv2.imread("shield.jpg")
faces = face_detector(img)

print("{} faces are detected.".format(len(faces)))

for f in faces:
    # 사각형 그리기
    print("left, top, right, bottom : ", f.left(), f.top(), f.right(), f.bottom())
    cv2.rectangle(img, (f.left(), f.top()), (f.right(), f.bottom()), (0, 0, 255), 2)

win = dlib.image_window()
win.set_image(img)
win.add_overlay(faces)
cv2.imwrite("output.jpg", img)

crop = img[f.top():f.bottom(), f.left():f.right()]
cv2.imwrite("cropped.jpg", crop)