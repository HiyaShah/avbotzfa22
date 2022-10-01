import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#https://stackoverflow.com/questions/57940737/how-to-mask-everything-in-an-image-except-for-a-specific-color-inside

image = cv2.imread('/Users/hiyashah/Desktop/avbotz/pathmarker_1.png')
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert to hsv scale
# lower = np.array([196, 65, 0], dtype="uint8") #idea - find max and min RBG values in the image!
# upper = np.array([255, 239, 213], dtype="uint8")
lower = np.array([6, 125, 138], dtype="uint8") #idea - find max and min RBG values in the image!
upper = np.array([25, 160, 180], dtype="uint8")
# lower = np.array([130, 130, 6], dtype="uint8") #idea - find max and min RBG values in the image!
# upper = np.array([164, 144, 10], dtype="uint8")
mask = cv2.inRange(image, lower, upper) #get a mask of only orange pixels

cnts  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(mask, cnts, (255,255,255)) #fill in mask (orange obj) with white
result = cv2.bitwise_and(original,original,mask=mask)



# plt.imshow(image)
# plt.show()
# plt.imshow(result)
# plt.show()
cv2.imshow('image', image)
cv2.imshow('mask', mask)
print("IM HERE")
cv2.imshow('result', result)
print("IM HERE 2")
cv2.waitKey()

# import numpy as np
# import cv2

# image = cv2.imread('/Users/hiyashah/Desktop/avbotz/1.png')
# original = image.copy()
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert to hsv color code
# lower = np.array([35, 0, 0], dtype="uint8")
# upper = np.array([131, 255, 185], dtype="uint8")
# mask = cv2.inRange(image, lower, upper)

# cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cv2.fillPoly(mask, cnts, (255,255,255))
# result = cv2.bitwise_and(original,original,mask=mask)

# cv2.imshow('mask', mask)
# cv2.imshow('result', result)
# cv2.waitKey()