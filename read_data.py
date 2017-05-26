#!/usr/bin/python3
"""
Author: Jiaqing Lin
E-mail: Jiaqing930@gmail.com
This program is used to convert video data to image data,
And make dataset for chainer framework.
"""

import numpy as np
import cv2 as cv
from PIL import Image
from glob import glob
import random
from chainer.datasets import tuple_dataset


# Convert video data to rgb image data frame by frame.
def create_spatial_images(rootPath='videos/*/*'):
    cv.useOptimized()
    folders = glob(rootPath)
    for path in folders:
        cnt_file = 0
        videoStream = cv.VideoCapture(path)
        while videoStream.isOpened():
            ret, frame = videoStream.read()
            if ret is True:
                resize = cv.resize(frame, (224, 224))
                img_path = path.replace('videos', 'rgb_images')
                img_path = img_path.replace('.avi', '_' + str(cnt_file) + '.jpg')
                cv.imwrite(img_path, resize)
                cnt_file = cnt_file + 1
            else:
                break


# Convert video data to multi-frame optical flow data.
def create_temporal_images(rootPath='videos/*/*'):
    new_frame_size = (224, 224)
    cv.useOptimized()
    folders = glob(rootPath)
    for path in folders:
        cnt_frame = 0
        videoStream = cv.VideoCapture(path)
        ret1, frame1 = videoStream.read()
        previous = cv.resize(frame1, new_frame_size)
        previous = cv.cvtColor(previous, cv.COLOR_BGR2GRAY)
        while True:
            ret2, frame2 = videoStream.read()

            if ret2 is False:
                break

            cnt_frame += 1
            if cnt_frame % 5 == 0:
                next = cv.resize(frame2, new_frame_size)
                next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
                # Use OpenCV function to get flow data from frame.
                flow = cv.calcOpticalFlowFarneback(previous, next, None, 0.5, 3, 15, 3, 5, 1.5,
                                                   cv.OPTFLOW_FARNEBACK_GAUSSIAN)
                x_direction_data = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX).astype('uint8')
                y_direction_data = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX).astype('uint8')

                img_path = path.replace('videos', 'flow_images')
                img_x_direction_path = img_path.replace('.avi', '_flow_' + str(cnt_frame) + '_x' + '.jpg')
                img_y_direction_path = img_path.replace('.avi', '_flow_' + str(cnt_frame) + '_y' + '.jpg')

                cv.imwrite(img_x_direction_path, x_direction_data)
                cv.imwrite(img_y_direction_path, y_direction_data)
                previous = next

        videoStream.release()
        cv.destroyAllWindows()


# Create a dataset of spatial model from rgb images, and normalize data [0. 255] to [0, 1].
def spatial_dataset():
    # Make label for each action.
    paths_labels = {'rgb_images/basketball/': 0,
                    'rgb_images/biking/': 1,
                    'rgb_images/diving/': 2,
                    'rgb_images/volleyball/': 3}

    full_data = []
    for path, label in paths_labels.items():
        imagelist = glob(path + '*')
        for imgPath in imagelist:
            full_data.append([imgPath, label])
    full_data = np.random.permutation(full_data)

    imageData = []
    labelData = []
    for path_label in full_data:
        img = Image.open(path_label[0])
        r, g, b = img.split()
        rData = np.asarray(np.float32(r) / 255.0)
        gData = np.asarray(np.float32(g) / 255.0)
        bData = np.asarray(np.float32(b) / 255.0)
        imgData = np.asarray([rData, gData, bData])
        imageData.append(imgData)
        labelData.append(np.int32(path_label[1]))

    threshold = np.int32(len(imageData) / 10 * 8)
    train = tuple_dataset.TupleDataset(imageData[0 : threshold], labelData[0 : threshold])
    test = tuple_dataset.TupleDataset(imageData[threshold : ], labelData[threshold : ])
    return train, test


# Create a dataset of tmporal model from images.
def temporal_dataset():
    # Make label for each action.
    paths_labels = {'flow_images/basketball/': 0,
                    'flow_images/biking/': 1,
                    'flow_images/diving/': 2,
                    'flow_images/volleyball/': 3}

    path_label = []
    for path, label in paths_labels.items():
        imagelist = glob(path + '*')
        for imgPath in imagelist:
            path_label.append([imgPath, label])

    flow_x = []
    flow_y = []
    imageData = []
    labelData = []
    for index in range(len(path_label)):
        if "_x" in path_label[index][0]:
            x_img = Image.open(path_label[index][0])
            flow_x.append(np.asarray(np.float32(x_img) / 255.0))
        if "_y" in path_label[index][0]:
            y_img = Image.open(path_label[index][0])
            flow_y.append(np.asarray(np.float32(y_img) / 255.0))
        if (index+1) % 20 == 0:
            imgData = np.asarray([flow_x[0], flow_x[1], flow_x[2], flow_x[3], flow_x[4], flow_x[5], flow_x[6], flow_x[7], flow_x[8], flow_x[9],
                                  flow_y[0], flow_y[1], flow_y[2], flow_y[3], flow_y[4], flow_y[5], flow_y[6], flow_y[7], flow_y[8], flow_y[9]])
            imageData.append(imgData)
            labelData.append(np.int32(path_label[index][1]))

    while len(labelData) > len(imageData):
        labelData.pop(random.randrange(len(labelData)))

    threshold = np.int32(len(imageData) / 10 * 8)
    train = tuple_dataset.TupleDataset(imageData[0: threshold], labelData[0: threshold])
    test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])
    return train, test


if __name__ == '__main__':
    # Create images from video.
    create_spatial_images()
    create_temporal_images()

