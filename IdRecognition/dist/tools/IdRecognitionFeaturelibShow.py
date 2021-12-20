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
import IdRecognitionResultShow as idr
import cv2


def putText(image, configDict, xt, yt):
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 255, 255)
    keys = ["uuid", "name", "gender", "age", "format", "width", "height"]
    for (i, key) in enumerate(keys):
        position = (xt, yt + 20 * (i + 1))
        image = cv2.putText(image, "%s:%s" % (key, configDict[key]), position,
                            font, 0.4, color, 1)
    return image


def searchFiles(filePath):
    i = 0
    while True:
        configFile = os.path.join(filePath, "%s.config" % str(i))
        if not os.path.exists(configFile):
            break
        configDict = idr.readCofigFile(configFile)
        binaryFile = os.path.join(filePath, "%s.bgr" % str(i))
        image = idr.readBinaryFile(binaryFile, int(configDict["height"]),
                                   int(configDict["width"]), 3)
        image = cv2.copyMakeBorder(image, 0, 20 * 7 + 10, 0, 0,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = putText(image, configDict, 0, int(configDict["height"]))
        outputJpgFile = os.path.join(filePath, "featureLib_%s.jpg" % str(i))
        cv2.imwrite("featureLib.jpg", image)
        cv2.imencode('.jpg', image)[1].tofile(outputJpgFile)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i += 1


if __name__ == '__main__':
    filePath = '...\\featureLib'
    searchFiles(filePath)
