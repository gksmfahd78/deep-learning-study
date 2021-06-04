import cv2
path = ""
image = cv2.imread(path)
print(image.shape)
cv2.imshow("Image", image)
cv2.waitKey