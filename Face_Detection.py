#author:chris
#email:christopherqiwei@gmail.com
#read trained model from a pickle file
import pickle
import cv2
import skimage
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, feature

class face_detection:
    def __init__(self, img, pickle_file):
        self.pickle_file = pickle_file
        self.img = img
       
    def _get_model(self):
        pickle_in = open(self.pickle_file, 'rb')
        model = pickle.load(pickle_in)
        return model
    
    def _input_img(self):
        image = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        if image.shape > (160, 140):
            x = (160/image.shape[0] + 140/image.shape[1])/2
        else:
            x = 1
        image = skimage.transform.rescale(image, x)
        return image
    
    def _get_patches_hog(self, image):
        indices, patches = zip(*self._sliding_window(image))
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        return patches_hog, indices
    def _sliding_window(self,image, patch_size=(62, 47),
                   istep=2, jstep=2, scale=1.0):
        Ni, Nj = (int(scale * s ) for s in patch_size)
        for i in range(0, image.shape[0] - Ni, istep):
            for j in range(0, image.shape[1] - Ni, jstep):
                patch = image[i:i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield(i, j), patch
    def predict(self):
        image = self._input_img()
        patches_hog, indices = self._get_patches_hog(image)
        model = self._get_model()
        labels = model.predict(patches_hog)
        print(label.sum())
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        Ni, Nj = (62, 47)
        indices = np.array(indices)
        for i, j in indices[labels == 1]:
            ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2,
                               facecolor='none'))
        return labels, image, indices
    
if __name__ == "__main__":
    x = face_detection('face.jpg','facedetection.pickle')
    labels, image,indices = x.predict()
    plt.show()
            
       