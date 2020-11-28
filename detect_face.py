#import dlib
import cv2
import os
import numpy as np

#step1: read the image

SZ=20
bin_n = 16
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
#step2: converts to gray image

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
#cv2.imshow('Image', gray)

def hog(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

image = cv2.imread('./test/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
if gray is None:
    raise Exception("we need the image from samples/data here !")
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]
deskewed = [list(map(gray,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in train_cells]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]


svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')
deskewed = [list(map(gray,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)

cv2.waitKey(0)
cv2.imwrite(os.path.join(path_write, '1.pgm'), gray )