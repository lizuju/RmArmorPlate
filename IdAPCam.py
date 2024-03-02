import cv2
import numpy as np
import cvzone
from kalman import KalmanFilter

cap = cv2.VideoCapture(1)

#kernel核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

#屏幕像素高度
whole_h = 1080

#裝甲板的一半寬度和一半高度
HALF_WIDTH = 70
HALF_HEIGHT = 30

# 定义物体的世界坐标，单位为mm
obj = np.array([
    [-HALF_WIDTH, -HALF_HEIGHT, 0],
    [HALF_WIDTH, -HALF_HEIGHT, 0],
    [HALF_WIDTH, HALF_HEIGHT, 0],
    [-HALF_WIDTH, HALF_HEIGHT, 0]
    ], dtype=np.float32)

# 相機內參和畸變系數
cam = np.array([[2.45473420e+03, 0, 9.66637900e+02], [0, 2.45226540e+03, 4.86822863e+02],
                [0, 0, 1]], dtype=np.float32)

# 初始化旋轉向量和平移向量
rVec = np.zeros((3, 1), dtype=np.float64)
tVec = np.zeros((3, 1), dtype=np.float64)

# 初始化Kalman滤波器
kf = KalmanFilter()

while True:
    success, img = cap.read()

    img_blur = cv2.GaussianBlur(img, (5, 5), 5)

    # 轉hsv
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # 選出紅色
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red, upper_red)
    mask_red = mask_red1 + mask_red2
    # 選出白色
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # 過濾白色，保留紅色
    mask_red_only = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_white))
    # #保留白色和紅色
    # mask_red_while = cv2.bitwise_and(mask_red, mask_white)

    opening = cv2.morphologyEx(mask_red_only, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(img, img, mask=opening)

    blur = cv2.GaussianBlur(res, (3, 3), 9)

    # 二值化
    # T = blur.mean()
    # t0 = blur[blur < T].mean()
    # t1 = blur[blur >= T].mean()
    # t = (t0 + t1) / 2
    # if abs(T - t) < 1:
    #     break
    # T = t
    # T = int(T)
    # print(T)
    thresh, dst = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)

    ero = cv2.erode(dst, kernel, iterations=2)
    dil = cv2.dilate(ero, kernel, iterations=2)

    # kernelInt核
    kernelInt = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dil2 = cv2.dilate(dil, kernelInt, iterations=4, borderType=cv2.BORDER_REFLECT_101)

    # 紅色遮罩
    red_mask = cv2.bitwise_and(dil2, res)

    dil_gray = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)

    # 膨漲，防止暗光
    kernel_mor = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dst_mor = cv2.morphologyEx(dil_gray, cv2.MORPH_CLOSE, kernel_mor)

    # 找輪廓
    contours, h = cv2.findContours(dst_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    width_array = []
    height_array = []
    point_array = []
    Midy_point = []
    Midx_point = []
    centers = []

    for contour in contours[:4]:
        (x, y, w, h) = cv2.boundingRect(contour)
        # 燈條條件
        is_valid = h/w > 2.8 and h/whole_h > 0.1 and w < 200
        if not is_valid:
            continue
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 7)
        width_array.append(w)
        height_array.append(h)
        point_array.append([x, y])
        Midy_point.append(y)
        Midx_point.append(x)

    point_near = [0, 0]
    # 兩燈條面積差最小值
    min = 10000
    # 裝甲板寬度
    W_REAL = 13
    # 攝相頭焦距
    F = 2446
    for i in range(len(width_array) - 1):
        for j in range(i+1, len(width_array)):
            # 兩燈條面積差
            value = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
            # 兩燈條在屏幕像素寬度
            w_pixel = abs(Midx_point[i] - Midx_point[j])
            # 攝相頭到裝甲板距離
            d = (W_REAL * F) / w_pixel
            d = abs(d)
            #print(d)
            # 裝甲板在屏幕像素面積
            #print(width_array[i] * height_array[i])
            #y = -270.082x + 58856.91
            # 圖像中檢測到的角點坐標
            pnts = np.array([[Midx_point[i], Midy_point[i]],
                             [Midx_point[j], Midy_point[j]],
                             [Midx_point[j], Midy_point[j]+height_array[j]],
                             [Midx_point[i], Midx_point[i]+height_array[i]]],
                            dtype=np.float32)

            # 裝甲板中心點坐標
            a = int((int(Midx_point[i]) + int(Midx_point[j])) / 2)
            b = int((int(Midy_point[i] + height_array[i] / 2.5) +
                     int(Midy_point[j] + height_array[j] / 2.5)) / 2)

            #裝甲板條件
            if value < min and abs(Midy_point[i] - Midy_point[j]) < 300 and \
                    (width_array[i] * height_array[i]) <-270.082*(d) + 78857 and \
                    (width_array[i] * height_array[i]) >-270.082*(d) - 38857:
                min = value
                point_near[0] = i
                point_near[1] = j

        try:
            rect1 = point_array[point_near[0]]
            rect2 = point_array[point_near[1]]
            point1 = [rect1[0] + width_array[point_near[0]] / 2, rect1[1]]
            point2 = [rect1[0] + width_array[point_near[0]] / 2,
                      rect1[1] + height_array[point_near[0]]]
            point3 = [rect2[0] + width_array[point_near[1]] / 2, rect2[1]]
            point4 = [rect2[0] + width_array[point_near[1]] / 2,
                      rect2[1] + height_array[point_near[1]]]
            x = np.array([point1, point2, point4, point3], np.int32)
            box = x.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 0, 255), 5)
            cvzone.putTextRect(img, f'Depth:{int(d)}cm', (100, 100), scale=4)
            # 进行位置解算
            success, rVec, tVec = cv2.solvePnP(obj, pnts, cam, None, rVec, tVec,
                                               False, cv2.SOLVEPNP_ITERATIVE)
            print("-----")
            print("rVec:", rVec)
            print("tVec:", tVec)
            print("-----")
            #畫出裝甲板中心坐標
            cv2.circle(img, (a, b), 10, (0, 20, 255), -1)
            centers.append(np.array([[a], [b]]))
            # #預測坐標
            predicted = kf.predict(a, b)
            predicted1 = kf.predict(predicted[0], predicted[1])
            cv2.circle(img, predicted1, 15, (20, 220, 0), 4)

        except:
            continue
    cv2.imshow("img", img)
    cv2.imshow("res", res)
    # cv2.imshow("img_sp", img_sp)
    cv2.imshow("blur", blur)
    cv2.imshow("dst", dst)
    cv2.imshow("ero", ero)
    cv2.imshow("dil", dil)
    cv2.imshow("dil", dil2)
    cv2.imshow("dil_gray", dil_gray)
    # cv2.imshow("dst_1", dst_1)
    # cv2.imshow('Rotated Image', rotated_img)
    cv2.imshow("dst_mor", dst_mor)
    # cv2.imshow("result", result)
    cv2.waitKey(1)
    if 0xFF == ord('q'):
        break