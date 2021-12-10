#!/usr/bin/env python
# coding=utf-8

"""
Copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import cv2
import subprocess
import numpy as np

keys = ["name", "gender", "age", "similarity", "faceId"]
resizeToW = 148  # ID图片缩放宽度
resizeToH = 207  # ID图片缩放高度
allTextH = 20  # 5行字，每行字高度20
spaceH = 10


def readCofigFile(configFile):
    config = {}
    with open(configFile, "r") as f:
        for line in f.readlines():
            lineLists = line.split('=')
            config[lineLists[0].strip()] = lineLists[1].strip()
    return config


def readBinaryFile(binaryFile, h, w, c):
    with open(binaryFile, mode='rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data = data.reshape(h, w, c)
        return data


def resizeIdImage(idImage):
    idImageH, idImageW, idImageC = idImage.shape
    aw = resizeToW / idImageW
    ah = resizeToH / idImageH
    a = min(aw, ah)
    idImageResize = cv2.resize(idImage, (0, 0), fx=a, fy=a,
                               interpolation=cv2.INTER_NEAREST)
    topOffset = int((resizeToH - idImageH * a) / 2)
    rightOffset = int((resizeToW - idImageW * a) / 2)
    image = cv2.copyMakeBorder(idImageResize, topOffset, resizeToH - topOffset,
                               rightOffset, resizeToW - rightOffset,
                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return image


def getFaceNum(filePath, fileName):
    faceNum = 0
    while True:
        configFileName = "%s_Face%s.config" % (fileName, str(faceNum))
        configFile = os.path.join(filePath, configFileName)
        if not os.path.exists(configFile):
            break
        faceNum += 1
    return faceNum


def putText(image, configDict, xt, yt):
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 255, 255)
    for (i, key) in enumerate(keys):
        position = (xt, yt + allTextH * (i + 1))
        image = cv2.putText(image, "%s:%s" % (key, configDict[key]), position,
                            font, 0.4, color, 1)
    return image


def putFaceBox(image, configDict):
    font = cv2.FONT_HERSHEY_DUPLEX
    p1 = (int(float(configDict['minx'])), int(float(configDict['miny'])))
    p2 = (p1[0] + int(float(configDict['width'])),
          p1[1] + int(float(configDict['height'])))
    color1 = (0, 255, 0)
    image = cv2.rectangle(image, p1, p2, color1, 2)
    position = (p1[0], p1[1] - 10)
    image = cv2.putText(image, 'faceId: ' + str(int(configDict["faceId"])),
                        position, font, 0.8, color1, 1)
    return image


def putLandmarks(image, configDict):
    for i in range(5):
        xy = (int(float(configDict["x%s" % str(i)])),
              int(float(configDict["y%s" % str(i)])))
        cv2.circle(image, xy, 1, (255, 0, 0), thickness=8)
    return image


def showImage(filePath, fileName):
    jpgFile = os.path.join(filePath, fileName + ".jpg")
    orgImage = cv2.imdecode(np.fromfile(jpgFile, dtype=np.uint8), -1)
    orgImageH = orgImage.shape[0]
    faceNum = getFaceNum(filePath, fileName)
    rowIdNum = int(orgImage.shape[1] / resizeToW)
    colIdNum = int(max(0, faceNum - 1) / rowIdNum + 1)
    bottomOffset = (resizeToH + 20 * len(keys) + spaceH) * colIdNum
    rightOffset = 0
    orgImage = cv2.copyMakeBorder(orgImage, 0, bottomOffset, 0, rightOffset,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    for iFace in range(faceNum):
        offsety = orgImageH + (resizeToH + allTextH * len(keys) + spaceH) * int(
            iFace / rowIdNum)
        configFileName = "%s_Face%s.config" % (fileName, str(iFace))
        configFile = os.path.join(filePath, configFileName)
        configDict = readCofigFile(configFile)
        idFileName = "%s_Face%s.bgr" % (fileName, str(iFace))
        idFile = os.path.join(filePath, idFileName)
        idImage = readBinaryFile(idFile, int(configDict["imgHeight"]),
                                 int(configDict["imgWidth"]), 3);
        idImageResize = resizeIdImage(idImage)
        i = iFace % rowIdNum
        orgImage[offsety:offsety + resizeToH,
        resizeToW * i: resizeToW * (i + 1)] = idImageResize[
                                              0:resizeToH,
                                              0:resizeToW]
        orgImage = putText(orgImage, configDict, resizeToW * i,
                           offsety + resizeToH)
        orgImage = putFaceBox(orgImage, configDict)
        orgImage = putLandmarks(orgImage, configDict)
    outputJpgFile = os.path.join(filePath, fileName + "_result.jpg")
    cv2.imwrite('result.jpg', orgImage)
    cv2.imencode('.jpg', orgImage)[1].tofile(outputJpgFile)
    # show on linux
    subprocess.run('eog result.jpg')
    # show on windows
    #cv2.imshow('image', orgImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def searchFiles(filePath, deep=0, fileName=""):
    if deep == 2:
        showImage(filePath, fileName)
        return
    dirs = os.listdir(filePath)
    for dir in dirs:
        if fileName == "":
            nextfileName = dir
        else:
            nextfileName = "%s_%s" % (fileName, dir)
        ff = os.path.join(filePath, dir)
        if os.path.isdir(ff):
            searchFiles(ff, deep + 1, nextfileName)


if __name__ == '__main__':
    filePath = '../result'
    searchFiles(filePath)
