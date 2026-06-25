import cv2

img = cv2.imread("cat.jpg")
print(img.shape)
cv2.imshow("Image", img)
cv2.waitKey(0)

grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", grayed)
cv2.waitKey(0)

resized = cv2.resize(img, (300, 150))
print(resized.shape)
cv2.imshow("Resized image", resized)
cv2.waitKey(0)

croped = img[:, 200:]
print(croped.shape)
cv2.imshow("Croped image", croped)
cv2.waitKey(0)

blured = cv2.GaussianBlur(croped, (9, 9), 0)
cv2.imshow("Blured image", blured)
cv2.waitKey(0)

edges = cv2.Canny(grayed, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

_, thresh = cv2.threshold(grayed, 127, 244, cv2.THRESH_BINARY)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contour_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contour image", contour_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
