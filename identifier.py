import cv2 as cv
import numpy as np


minArea = 100000

def biggestArea(image, threshold):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)

    # Erode and dilate
    cv.erode(threshold, kernel, 2)
    cv.dilate(threshold, kernel, 2)

    contours, hierarchy = cv.findContours(threshold,
                                          cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    largestContour = np.array([[]])
    area = 0

    # if contours have been detected, draw them
    if len(contours) > 0:
        # cv.drawContours(image, contours, -1, 255, 2)

        largestContour = max(contours, key=cv.contourArea)

        area = cv.contourArea(largestContour)

    return [area, largestContour]


def runPipeline(image, llrobot):
    imgHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # convert the hsv to a binary image by removing pixels
    # not in the HSV min/max values
    coneThresh = cv.inRange(imgHSV, (20, 48, 82), (59, 255, 255))
    cubeThresh = cv.inRange(imgHSV, (100, 100, 50), (140, 255, 255))

    [coneArea, biggestCone] = biggestArea(image, coneThresh)
    [cubeArea, biggestCube] = biggestArea(image, cubeThresh)

    # print(str(coneArea) + " " + str(cubeArea))

    # initialize empty array to send data to bot
    llpython = [0]

    largestContour = np.array([[]])

    if coneArea > cubeArea and coneArea > minArea:
        llpython[0] = 1
        largestContour = biggestCone
    elif cubeArea > coneArea and cubeArea > minArea:
        llpython[0] = 2
        largestContour = biggestCube

    # print(str(cubeArea) + " " + str(coneArea))

    cv.putText(image, str(llpython[0]), (200, 200), cv.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 10, cv.LINE_AA)

    return largestContour, image, llpython