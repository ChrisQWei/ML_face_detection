# ML_face_detection
Face Detection by SVM
The facedetection.pickle has been trained
by SVM, so you can use it immediately.

How to use it :
from Face_Detection import face_detection
x = face_detection('face.jpg','facedetection.pickle')
labels, image,indices = x.predict()
plt.show()

The picture is random picture from internet.
Enjoy!
If you have any questions or whatever, please email me : christopherqiwei@gmail.com
