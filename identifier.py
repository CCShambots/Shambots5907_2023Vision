import cv2 as cv
import numpy as np


minArea = 1000

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
    coneThresh = cv.inRange(imgHSV, (25, 100, 0), (35, 255, 255))
    cubeThresh = cv.inRange(imgHSV, (100, 100, 50), (200, 255, 255))

    [coneArea, biggestCone] = biggestArea(image, coneThresh)
    [cubeArea, biggestCube] = biggestArea(image, cubeThresh)

    # print(str(coneArea) + " " + str(cubeArea))

    # initialize empty array to send data to bot
    llpython = ["none"]

    largestContour = np.array([[]])

    if coneArea > cubeArea and coneArea > minArea:
        llpython[0] = "cone"
        largestContour = biggestCone
    elif cubeArea > coneArea and cubeArea > minArea:
        llpython[0] = "cube"
        largestContour = biggestCube

    # cv.imshow('Stream', image)
    cv.putText(image, llpython[0], (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    return largestContour, image, llpython


# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('both.mkv')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        [c, i, p] = runPipeline(frame, 0)
        cv.drawContours(frame, [c], -1, 255, 2)
        cv.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
