import collections
import time

import cv2
import numpy as np

MODE = "MPI"

if MODE is "COCO":
    protoFile = "D:\IDA\PROJECT\Video_Image_Pose\coco\pose_deploy_linevec.prototxt"
    weightsFile = "D:\IDA\PROJECT\Video_Image_Pose\coco\pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                  [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE is "MPI":
    protoFile = r"D:\IDA\PROJECT\Video_Image_Pose\mpi\pose_deploy_linevec.prototxt"
    weightsFile = r"D:\IDA\PROJECT\Video_Image_Pose\mpi\pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                  [14, 11], [11, 12], [12, 13]]

userImageInput = cv2.VideoCapture(r"D:\IDA\PROJECT\Video_Image_Pose\Video_Pose\inputVideo\input_1.mp4")
frame_width = int(userImageInput.get(3))
frame_height = int(userImageInput.get(4))
fps = int(userImageInput.get(5))

img = r"D:\IDA\PROJECT\Video_Image_Pose\White.jpg"

frameWhite = cv2.imread(img)
frameWidth1 = frameWhite.shape[1]
frameHeight1 = frameWhite.shape[0]

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

out = cv2.VideoWriter(r'D:\IDA\PROJECT\Video_Image_Pose\Video_Pose\outputVideo\1.mp4', fourcc, 30, (frameWidth1, frameHeight1))
out1 = cv2.VideoWriter(r'D:\IDA\PROJECT\Video_Image_Pose\Video_Pose\outputVideo\2.mp4', fourcc, fps, (frame_width, frame_height))
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()
inWidth = 368
inHeight = 368
while True:

    ret, frame = userImageInput.read()
    if not ret:
        break
    img = r"D:\IDA\PROJECT\Video_Image_Pose\White.jpg"

    frameWhite = cv2.imread(img)
    frameWidth1 = frameWhite.shape[1]
    frameHeight1 = frameWhite.shape[0]

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]
    points = []
    points1 = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original RunningImage
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        x1 = (frameWidth1 * point[0]) / W
        y1 = (frameHeight1 * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frameWhite, (int(x1) + 70, int(y1)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            points.append((int(x), int(y)))
            points1.append((int(x1) + 70, int(y1)))
        else:
            points.append(None)
            points1.append(None)

    dictAngle = {'leftHand': [5, 6, 7], 'leftLeg': [11, 12, 13], 'rightHand': [2, 3, 4], 'rightLeg': [8, 9, 10]}
    dictAngle = collections.OrderedDict(sorted(dictAngle.items()))

    usernameDict = []
    userangleDict = []
    heightAngle = [350, 370, 390, 410]
    j = 0

    for i in dictAngle:
        dictPoint1 = np.array(points[dictAngle[i][0]])
        dictPoint2 = np.array(points[dictAngle[i][1]])
        dictPoint3 = np.array(points[dictAngle[i][2]])

        if str(dictPoint1) != 'None' and str(dictPoint2) != 'None' and str(dictPoint3) != 'None':

            ba = dictPoint1 - dictPoint2
            bc = dictPoint3 - dictPoint2

            tup1 = points1[dictAngle[i][1]]

            if i is 'leftHand' or i is 'leftLeg':
                pointAngle = (tup1[0] + 15, tup1[1] + 18)

            if i is 'rightHand' or i is 'rightLeg':
                pointAngle = (tup1[0] - 50, tup1[1])
            # print(point)

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            ang = str(np.degrees(angle))
            angleFloat = float(ang)
            ang1 = round(angleFloat, 2)
            # print(ang)
            cv2.putText(frameWhite, "Right Side", (15, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, "Left Side", (500, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, "User-Image", (700, 420), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frameWhite, str(ang1), pointAngle, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frameWhite, str(i + ": "), (15, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(frameWhite, str(ang), (90, heightAngle[j]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1,
                        cv2.LINE_AA)

            usernameDict.append(i)
            userangleDict.append(ang)
        else:
            usernameDict.append(i)
            userangleDict.append('0')
        j += 1
    # print(userangleDict, "aadmin")
    # print(dictAngle,"admin.......")

    for pair in POSE_PAIRS:

        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.line(frameWhite, points1[partA], points1[partB], (0, 255, 255), 2)

    userDict = dict(zip(usernameDict, userangleDict))

    out1.write(frame)
    out.write(frameWhite)
    AngleDict = str(
        userDict['rightHand'] + "," + userDict['leftLeg'] + "," + userDict['leftHand'] + "," + userDict['rightLeg'])
