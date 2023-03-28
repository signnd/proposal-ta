from turtle import width
import cv2
import matplotlib.pyplot as plt
import numpy as np

# import image
image = cv2.imread("images/aksara bali e2.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)

# erosion if needed, adjust kernels as required
#kernel = np.ones((3,3),np.uint8)
#erosion = cv2.erode(binary, kernel, iterations = 1)

# dilation
kernel0 = np.ones((101,59),np.uint8)
dil = cv2.dilate(binary, kernel0, iterations = 1)

# additional img processing if needed
#kernel2 = np.ones((33,29),np.uint8)
#dilation = cv2.dilate(erosion, kernel2, iterations = 1)
plt.imshow(dil)
plt.show()

# find contours from the thresholded image
#contours, hierarchy = cv2.findContours(cv2.morphologyEx(dilation, cv2.MORPH_OPEN, np.ones((2,2))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
contours, hierarchy = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# nh, nw = image.shape[:2]
i = 0
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    #if h >= 0.1 * nh:
    # preview image before cropping, bounding box will be shown on detected objects
    #cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,0), 10, cv2.LINE_AA)

    # crop objects
    # when cropping, make sure to comment line 37 so cropped images won't show bounding boxes
    cv2.imwrite('images/new3/img_{}.jpg'.format(i), image[y:y+h,x:x+w])
    i += 1
plt.imshow(image)
plt.show()
