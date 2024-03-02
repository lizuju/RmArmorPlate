import cv2
import numpy as np

cap = cv2.VideoCapture(1)

def P4P():
    HALF_WIDTH = 70
    HALF_HEIGHT = 30

    # 定义物体的世界坐标，单位为mm
    obj = np.array([
        [-HALF_WIDTH, -HALF_HEIGHT, 0],
        [HALF_WIDTH, -HALF_HEIGHT, 0],
        [HALF_WIDTH, HALF_HEIGHT, 0],
        [-HALF_WIDTH, HALF_HEIGHT, 0]
    ], dtype=np.float32)

    #圖像中檢測到的角點坐標
    pnts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    #相機內參和畸變系數
    cam = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dis = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    #初始化旋轉向量和平移向量
    rVec = np.zeros((3, 1), dtype=np.float64)
    tVec = np.zeros((3, 1), dtype=np.float64)

    # while True:
    #     success, img = cap.read()

    #進行位置解算
    success, rVec, tVec = cv2.solvePnP(obj, pnts, cam, dis, rVec, tVec, False, cv2.SOLVEPNP_ITERATIVE)
