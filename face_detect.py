''' usage :
1- python3 face_detect.py -v videos/test1.mkv
2- python3 face_detect.py -i images/img3.jpg
'''
import argparse as arg
import time
import cv2
import numpy as np
from skin_seg import *
class Face_Detector():
	def __init__(self,skin_detect):
		"skin_detect is an object from skin_seg file"
		self._skin_detect = skin_detect
	@property
	def skin_detect(self):
		"set skin_detect to be an immutable field/property"
		return self._skin_detect
	def Detect_Face_Img(self,img,size1,size2):
		'''this method implements the skin detection algorithm to perform a face detection in a given image.
		-inputs: 
		img : BGR image (numpy array)
		size1 : the lower size of a rectangle/face(min size) (type tuple)
		size2 : the upper size of a rectangle/face(max size) (type tuple)
		-output:
		a numpy array with all faces coordinates in a picture.
		'''
		#get the RGB_H_CbCr representation of the image(for more info, please refer to skin_seg.py)
		skin_img = self._skin_detect.RGB_H_CbCr(img,False)
		contours, hierarchy = cv2.findContours(skin_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		#cv2	.drawContours(img, contours, -1, (0,255,0), 1)
		#cv2.imshow("faces",img)
		#if cv2.waitKey(0) & 0xFF == ord("q"):
		#	sys.exit(0)
		rects = []
		for c in contours:
			# get the bounding rect
			x, y, w, h = cv2.boundingRect(c)
			# draw a green rectangle to visualize the bounding rect
			if (w > size1[0] and h > size1[1]) and (w < size2[0] and h < size2[1]):
				#pinhole distance
				Distance1 = 11.5*(img.shape[1]/float(w))
				#camera distance
				Distance2 = 15.0*((img.shape[1] + 226.8)/float(w))
				print("\npinhole distance = {:.2f} cm\ncamera distance = {:.2f} cm".format(Distance1,Distance2))
				print("Width = {} \t Height = {}".format(w,h))
				rects.append(np.asarray([x,y,w,w*1.25], dtype=np.uint16))
		return rects
	def Detect_Face_Vid(self,vid,size1,size2,scale_factor = 3):
		'''this method implements the skin detection algorithm to perform a face detection in a given video file.
		-inputs: 
		vid : video object 
		size1 : the lower size of a rectangle/face(min size) (type tuple)
		size2 : the upper size of a rectangle/face(max size) (type tuple)
		scale_factor : parameter for scaling down the image for a better frame rate
		-output:
		void
		'''		
		while True:
			start =time.time()
			(grabbed, img) = vid.read()
			if not grabbed:
				break
			#get the frame rate
			fps = vid.get(cv2.CAP_PROP_FPS)
			print("\nRecording at {} frame/sec".format(fps))
			Image = cv2.resize(img, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
			rects = self.Detect_Face_Img(Image,size1,size2)
			for i,r in enumerate(rects):
				# Scale back up face locations since the frame we detected in was scaled to 1/10 size
				x0,y0,w,h = r
				x0 *= scale_factor
				y0 *= scale_factor
				w *= scale_factor
				h *= scale_factor
				cv2.rectangle(img, (x0,y0),(x0+w,y0+h),(0,255,0),1)
				font = cv2.FONT_HERSHEY_SIMPLEX
			stop = time.time()
			# f = 30 frame/sec
			# T = 1/30 sec/frame
			# T = 0.032
			#frame = cv2.resize(frame, dim, interpolation =  cv2.INTER_AREA)
			time.sleep(abs((1/fps - (stop - start))))
			cv2.imshow('faces', img)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
		vid.release()
def Arg_Parser():
	Arg_Par = arg.ArgumentParser()
	Arg_Par.add_argument("-i", "--image",
					help = "relative/absolute path of the image file")
	Arg_Par.add_argument("-v", "--video",
					help = "relative/absolute path of the recorded video file")
	arg_list = vars(Arg_Par.parse_args())
	return arg_list
def open_img(arg_):
	mg_src = arg_["image"]
	img = cv2.imread(mg_src)
	img_arr = np.array(img, 'uint8')
	return img_arr	 
def open_vid(arg_):
	vid_src = "videos/video1.mkv"
	vid = cv2.VideoCapture(arg_["video"])
	return vid
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Please give me a file :Image/video !!!")
		print("\n Try Again, For more info type --help to see available options")
		sys.exit(0)
	in_arg = Arg_Parser()
	skin_detect = Skin_Detect()
	size1 = (40,40)
	size2 = (300,400)
	scale_factor = 3
	Face_Detect = Face_Detector(skin_detect)
	if in_arg["image"] != None:
		img = open_img(in_arg)
		rects = Face_Detect.Detect_Face_Img(img,size1,size2)
		print(rects)
		for i,r in enumerate(rects):
			x,y,w,h = r
			cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
		cv2.imshow("faces",img)
		if cv2.waitKey(0) & 0xFF == ord("q"):
			sys.exit(0)
	if in_arg["video"] != None:
		vid = open_vid(in_arg)
		Face_Detect.Detect_Face_Vid(vid,size1,size2,scale_factor)

