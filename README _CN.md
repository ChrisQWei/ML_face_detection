# ML_face_detection
使用SVM进行人脸识别.
'facedetection.pickle' 已经进行了SVM训练，
所以可以直接使用。
调用方式：
from Face_Detection import face_detection
x = face_detection('face.jpg','facedetection.pickle')
labels, image,indices = x.predict()
plt.show()

图片来源于网络。
Enjoy！
如有问题请email我:christopherqiwei@gmail.com