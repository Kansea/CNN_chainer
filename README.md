Author: Jiaqing Lin
E-mail: jiaqing930@gmail.com
Code for paper”Tow-Stream Convolutional Networks for Action Recognition in Videos”

In this folder includes video data and source code.
Folders:	
	1. “videos” folder has 4 classes video, basketball, biking, diving, and volleyball,
	    each class includes 2 video file as .avi format (size is about 150 - 300 KB).
	2. “rgb_images” and “flow_images” are prepared to store converted images.
	3. The training log file is stored in “spatial_result” and “temporal_result” folders.
	    Because training and test data are too small, so result is not good enough.
Files:
	1. “labels.txt” is stored each class label.
	2. “result.png” is shown snapshot when running code in terminal.
Code:
	1. “read_data.py” is used to convert video data to image data for training models.
	2. “spatial_model.py” includes spatial stream model, training, and evaluating.
	3. “temporal_model.py” includes temporal stream model, training, and evaluating.
	
	Each layer parameters is set by paper “Two-Stream Convolutional Networks for
	Action Recognition in Videos” and other related papers, but parameters of training
	is set by myself.

Library version:
	Python version 3.6.1
	Chainer version 1.23.0
	OpenCV version 3.2.0
	Numpy version 1.12.1
	Video dataset is download from youtube video dataset

Running code step:
	1. : > python3 read_data.py	(convert video data to each image data)
	2. : > python3 spatial_model.py (training and evaluating)
	3. : > python3 temporal_model.py (training and evaluating)
