import cv2 as cv
import numpy as np
import math

cap = cv.VideoCapture("cone.mkv")

if not cap.isOpened():
    exit

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # cv.imshow('Stream', frame)

        imgHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        imgThresh = cv.inRange(imgHSV, (25, 100, 0), (35, 255, 255))

        # cv.imshow("treshold", imgThresh)

        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((5, 5), np.uint8)

        # Erode and dilate
        cv.erode(imgThresh, kernel, 2)
        cv.dilate(imgThresh, kernel, 2)

        contours, hierarchy = cv.findContours(imgThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


        # imDisplay = imgThresh.copy()
        frameCopy = frame.copy()

        c = max(contours, key=cv.contourArea)

        box = cv.minAreaRect(c)
        rect = cv.boxPoints(box)
        rect = np.int0(rect)

        id = 0
        for point in rect:
            cv.putText(frameCopy, str(id), (point[0], point[1]), cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 2, cv.LINE_AA)
            id += 1

        

        # print(box[0])

        # originPoint = rect[0]
        # endPoint = rect[3]
        #
        # point = [endPoint[0]-originPoint[0], endPoint[1]-originPoint[1]]
        # angle = math.atan(point[0]/point[1]) * 180 / math.pi

        cv.putText(frameCopy, str(box[2]), (150, 250), cv.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 2, cv.LINE_AA)
        cv.drawContours(frameCopy, c, -1, (255, 0, 0), 3)
        cv.drawContours(frameCopy, [rect], 0, (0, 0, 255), 3)

        cv.imshow('Contours', frameCopy)

        # Press Q
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()

cv.destroyAllWindows()


def midpoint(point1, point2):
    return [(point1[0] + point2[0])/2, (point1[1] + point2[1])/2]
