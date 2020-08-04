# encoding ：UTF-8

'''
利用opencv自带的基于深度学习训练的函数来做人脸识别，准确率比Haar cascades要高
'''
# import os
# g = os.walk(r"E:\PycharmProjects\untitled\tensorflow-serving-yolov3-master\docs\images")
# print(os.listdir(r"E:\PycharmProjects\untitled\tensorflow-serving-yolov3-master\docs\images"))
# n = 0
# for path, d, filelist in g:
#     for filename in filelist:
#         f = os.path.join(path, filename)
#         print(f)
#     if not os.path.exists("./JsonToXml/xmlDataset/ImageSets/Main"):
#         os.makedirs("./JsonToXml/xmlDataset/ImageSets/Main")


import numpy as np
import cv2

# 定义相关的路径参数
modelPath = "H:/PycharmProjects/untitled/qt_jiemian/deploy.prototxt.txt"
weightPath = "H:/PycharmProjects/untitled/qt_jiemian/res10_300x300_ssd_iter_140000.caffemodel"
confidence = 0.5  # 置信度参数，高于此数才认为是人脸，可调


def face_detector(image):
    net = cv2.dnn.readNetFromCaffe(modelPath, weightPath)

    # 输入图片并重置大小符合模型的输入要求
    (h, w) = image.shape[:2]  # 获取图像的高和宽，用于画图
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # ===============================blobFromImage待研究=====================

    net.setInput(blob)
    detections = net.forward()  # 预测结果

    # 可视化：在原图加上标签和框
    for i in range(0, detections.shape[2]):
        # 获得置信度
        res_confidence = detections[0, 0, i, 2]

        # 过滤掉低置信度的像素
        if res_confidence > confidence:
            # 获得框的位置
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 在图片上写上标签
            text = "{:.2f}%".format(res_confidence * 100)
            # 如果检测脸部在左上角，则把标签放在图片内，否则放在图片上面
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            # print(startX)
            # print(endX)
            # print(startY)
            # print(endY)
            # print('sssssssssssssssss')
            if startX - 10 > 0 and startY - 10 > 0 and endX + 10 < w and endY + 10 < h:
                roi = image[startY - 10:endY + 10, startX - 10:endX + 10]
                (h_roi, w_roi) = roi.shape[:2]
                for i in range(h_roi):
                    for j in range(w_roi):
                        roi[i, j] = (0, 0, 255)  # 这里可以处理每个像素点
                cv2.imwrite("H:/PycharmProjects/meme.jpg", image)
                cv2.waitKey(0)

    # cv2.putText(image, text, (startX, y),
    # 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    resImage = image
    return resImage


if __name__ == '__main__':
    image = "H:/PycharmProjects/untitled/tensorflow-serving-yolov3-master/docs/meme.jpg" 
    image = cv2.imread(image)

    resImage = face_detector(image)

