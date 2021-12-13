import numpy as np
import cv2 as cv
from numpy.lib.arraypad import pad
import matplotlib.pyplot as plt
import time

print(cv.__version__)
print(np.__version__)

mask = np.ones((5,5)) / 25

img = cv.imread('../girl.png', 0)

img_numpy = np.asarray(img)

start = time.time()

output = conv2d(img_numpy, mask)

end = time.time()

print(str(end-start) + ' seconds')

plt.figure(1)
plt.imshow(img,cmap='gray')

plt.figure(2)
plt.imshow(output,cmap='gray')

