from skimage import morphology
import cv2
import os

dir = "infers"
for img_name in os.listdir(dir):
    img = cv2.imread(os.path.join(dir, img_name), -1)
    img = morphology.remove_small_objects(img, min_size=10)
    cv2.imwrite(os.path.join(dir, img_name), img)
