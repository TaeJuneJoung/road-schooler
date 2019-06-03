import cv2 as cv

origin_img = cv.imread('June/2016080401832_1.jpg')
gray_img = cv.cvtColor(origin_img, cv.COLOR_BGR2GRAY)

face_cascade = cv.CascadeClassifier('June/haarcascade_frontalface_alt.xml')
detected_faces = face_cascade.detectMultiScale(gray_img)

for col, row, width, height in detected_faces:
    cv.rectangle(
        origin_img,
        (col, row),
        (col + width, row + height),
        (0, 255, 0),
        2
    )

cv.imshow('Image', origin_img)
cv.waitKey(0)
cv.destroyAllWindows()
