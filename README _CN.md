# ML_face_detection
ʹ��SVM��������ʶ��.
'facedetection.pickle' �Ѿ�������SVMѵ����
���Կ���ֱ��ʹ�á�
���÷�ʽ��
from Face_Detection import face_detection
x = face_detection('face.jpg','facedetection.pickle')
labels, image,indices = x.predict()
plt.show()

ͼƬ��Դ�����硣
Enjoy��
����������email��:christopherqiwei@gmail.com