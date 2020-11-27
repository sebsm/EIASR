#import dlib
import cv2
import os

#step1: read the image
image = cv2.imread('./test/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')

#step2: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
path_read = 'E:/EIASR/PROJEKT/test'
path_write = 'E:/EIASR/PROJEKT/result'
#step3: get HOG face detector and faces
#hogFaceDetector = dlib.get_frontal_face_detector()
#faces = hogFaceDetector(gray, 1)

#step4: loop through each face and draw a rect around it
# for (i, rect) in enumerate(faces):
#     x = rect.left()
#     y = rect.top()
#     w = rect.right() - x
#     h = rect.bottom() - y
#     #draw a rectangle
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#step5: display the resulted image
#cv2.imshow('Image', image)
cv2.imshow('Image', gray)
cv2.waitKey(0)
cv2.imwrite(os.path.join(path_write, '1.pgm'), gray )