import cv2
import numpy as np 
from PIL import Image, ImageChops
 
# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png
img = cv2.imread('7.jpg')

im = Image.fromarray(img)
bg = Image.new(im.mode, im.size, (255, 255, 255))
diff = ImageChops.difference(im, bg)
diff = ImageChops.add(diff, diff, 2.0, -100)
bbox = diff.getbbox()
if bbox:
    im = im.crop(bbox)
    img = np.array(im)

height, width, _ = img.shape

img = img[1:height - 3, 1:width - 3]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

ret, threshed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
high = ret
low = int(high * 0.5)

#-- Edge detection -------------------------------------------------------------------
edges = cv2.Canny(gray, low, high)
edges = cv2.dilate(edges, None)
edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
contour_info = []
_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 10000:
        print cv2.isContourConvex(c)
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
print [c[2] for c in contour_info]
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

x, y, w, h = cv2.boundingRect(max_contour[0])
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
cv2.drawContours(img, [c[0] for c in contour_info], -1, (255, 255, 0), 1)
 
cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()