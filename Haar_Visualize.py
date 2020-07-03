import cv2
from xml.dom import minidom
import numpy as np
from scipy.signal import convolve2d as conv
import matplotlib.pyplot as plt
import sys
if __name__ == "__main__":
	# path to the cascade file 
	cascade_path = "Haar_Cascades/haarcascade_frontalface_default.xml"
	# open and parse the xml file
	doc_Tree = minidom.parse (cascade_path)
	width = doc_Tree.getElementsByTagName ("width")[0]
	print ("Node Name :",width.nodeName)
	width = int(width.firstChild.data)
	print ("Node Value :",width)
	height = doc_Tree.getElementsByTagName ("height")[0]
	print ("Node Name :",height.nodeName)
	height = int(height.firstChild.data)
	print ("Node Value :",height)
	root_node = doc_Tree.documentElement
	# Haar feature
	rects = root_node.getElementsByTagName ("rects")
	print ("root node :",root_node.nodeName)
	# create a feature matrix
	feat_mat = []
	count = 1
	# loop through each rect
	for rect in rects:
		kernel = np.zeros((height, width))
		for i,line in enumerate(rect.getElementsByTagName ('_')):
			line_list = line.childNodes[0].data.strip().split(" ")
			x1, y1, x2, y2 = map(int, line_list[:4])
			x1, x2 = min(x1, x2), max(x1, x2)
			y1, y2 = min(y1, y2), max(y1, y2)
			c = float(line_list[4])
			kernel[y1:y2+1, x1:x2+1] = c
			#print(line_list)
		feat_mat.append(kernel)
		#plt.imshow(kernel, cmap="gray", interpolation='none') # Plot the image, turn off interpolation
		#plt.pause(0.01)
		#plt.clf()
		count+=1
	print(count)
	#print(feat_mat[0])
	#plt.imshow(feat_mat[0], cmap="gray", interpolation='none') # Plot the image, turn off interpolation
	#plt.show() # Show the image window
	weakClassifiers = root_node.getElementsByTagName ("weakClassifiers")
	stageThresholds = root_node.getElementsByTagName ("stageThreshold")
	stages_list = []
	for weakClassifier,threshold in zip(weakClassifiers,stageThresholds):
		thresholds = float(threshold.childNodes[0].data.strip())
		classifiers = []
		#print(thresholds)
		for internalNode,leafValue in zip(weakClassifier.getElementsByTagName ('internalNodes'),
			weakClassifier.getElementsByTagName ('leafValues')):
			internal_node = internalNode.childNodes[0].data.strip().split(" ")
			feat_num = int(internal_node[2])
			feat_thresh = float(internal_node[3])
			leafs = leafValue.childNodes[0].data.strip().split(" ")
			lower_leaf = float(leafs[0])
			upper_leaf = float(leafs[1])
			classifiers.append([feat_num, feat_thresh, lower_leaf, upper_leaf])
		stages_list.append([threshold, classifiers])
	#print(stages_list[0][1])
	image = cv2.imread("images/img3.jpg", 0)
	#image = cv2.resize(image, (0,0), fx=1/2, fy=1/2)
	image_height, image_width = image.shape
	plt.figure()
	count = 0
	for stage in stages_list:

		image1 = image.copy()

		for classifier in stage[1]:
			feat_num, feat_thresh, lower_leaf, upper_leaf = classifier
			print(classifier)
			act_map = conv(image, feat_mat[feat_num], mode="valid")
			if upper_leaf > lower_leaf:
				act_map[act_map < feat_thresh] = 0
			else:
				act_map[act_map > feat_thresh] = 0

			k = 6
			flat_act_map = act_map.flatten()
			top_indices = np.argpartition(flat_act_map, kth=-k, axis=-1)[-k:]
			top_indices = top_indices[flat_act_map[top_indices] > 0]

			for top_index in top_indices:
				i, j = np.unravel_index(top_index, act_map.shape)
				image_part = image1[i:i+height, j:j+width].astype(np.uint8)
				rect = np.ones(image_part.shape, dtype=np.uint8) * 255
				alpha = 0.4
				image2 = image1.copy()
				res = cv2.addWeighted(image_part, alpha, rect, 1 - alpha, gamma = 0)
				image1[i:i+height, j:j+width] = res
				image2[i:i+height, j:j+width] = feat_mat[feat_num]
				image2 = cv2.resize(image2, (0,0), fx=2, fy=2)
				cv2.imshow('video', image2)
				k = cv2.waitKey(10) & 0xff  # 'ESC' for Exit
				if k == 27:
					break
		count+=1
		print(count)
		#plt.imshow(image_copy, cmap="gray")
		#plt.pause(0.001)
		image2 = image1.copy()
		image2 = cv2.resize(image2, (0,0), fx=4, fy=4)
		cv2.imwrite("pic" + '.' + str(count) + ".jpg " ,image2)
		
