import cv2
import numpy as np

img = cv2.imread("Resources/img4.jpeg")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

whole_h = 1080

class Split():
    def img_b(self):
        b, g, r = cv2.split(img)
        return b

    def img_r(self):
        b, g, r = cv2.split(img)
        return r

split = Split()
img_sp = split.img_r()

while True:
    blur = cv2.GaussianBlur(img_sp, (3, 3), 9)

    thresh, dst = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    ero = cv2.erode(dst, kernel, iterations=2)
    dil = cv2.dilate(ero, kernel, iterations=2)

    contours, h = cv2.findContours(dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    width_array = []
    height_array = []
    point_array = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        is_valid = h/w > 2.4 and h/whole_h > 0.1
        if not is_valid:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 7)
        width_array.append(w)
        height_array.append(h)
        point_array.append([x, y])

    point_near = [0, 0]
    min = 10000
    for i in range(len(width_array) - 1):
        for j in range(i+1, len(width_array)):
            value = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
            if value < min:
                min = value
                point_near[0] = i
                point_near[1] = j
        try:
            rect1 = point_array[point_near[0]]
            rect2 = point_array[point_near[1]]
            point1 = [rect1[0] + width_array[point_near[0]] / 2, rect1[1]]
            point2 = [rect1[0] + width_array[point_near[0]] / 2, rect1[1] + height_array[point_near[0]]]
            point3 = [rect2[0] + width_array[point_near[1]] / 2, rect2[1]]
            point4 = [rect2[0] + width_array[point_near[1]] / 2, rect2[1] + height_array[point_near[1]]]
            x = np.array([point1, point2, point4, point3], np.int32)
            box = x.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)
        except:
            continue
        cv2.imshow("img", img)
        cv2.imshow("r", img_sp)
        cv2.waitKey(0)
