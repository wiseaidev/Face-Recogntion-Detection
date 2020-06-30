import sys
import cv2
import os
import time
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt
class Train_Model():
	def __init__(self,face_cascade,right_eye_cascade,left_eye_cascade,lbph_var):
		self._Face_Cascade = cv2.CascadeClassifier(face_cascade)
		self._Right_Eye_Cascade = cv2.CascadeClassifier(right_eye_cascade)
		self._Left_Eye_Cascade = cv2.CascadeClassifier(left_eye_cascade)
		self.path_exists("dataset/")
		self.recognizer = cv2.face.LBPHFaceRecognizer_create(lbph_var[0], lbph_var[1], lbph_var[2], lbph_var[3])
	def path_exists(self,path):
		dir = os.path.dirname(path)
		if not os.path.exists(dir):
			os.makedirs(dir)
	def FileRead(self):
		NAME = []  
		with open("users_name.txt", "r") as f :                      
			for line in f:         
				NAME.append (line.split(",")[1].rstrip())
		return NAME        
	def Add_User(self):
		Name = input('\n[INFO] Please Enter a user name and press <return> ==> ')
		Info = open("users_name.txt", "a+")
		ID = len(open("users_name.txt").readlines(  )) + 1
		Info.write(str(ID) + "," + Name + "\n")
		print ("\n[INFO] This Person has ID = " + str(ID))
		Info.close()
		return ID
	def getImagesAndLabels(self,path):
		imagePaths = [os.path.join(path,f) for f in os.listdir (path)]
		faceSamples = []
		ids = []
		for imagePath in imagePaths:
			img = cv2.imread(imagePath,0)
			img_numpy = np.array (img, 'uint8' )
			id = int (os.path.split (imagePath) [- 1] .split ( "." ) [1])
			faceSamples.append (img_numpy)
			ids.append (id)
		return faceSamples, ids
	def train(self,path,file_name):
		print ( "\n[INFO] Face training has been started, please wait a moment..." )
		# slight delay
		time.sleep(1)
		faces, ids = self.getImagesAndLabels (path)
		self.recognizer.update (faces, np.array (ids))
		# Saving the model
		self.recognizer.write (file_name)
		print ( "\n[INFO] {0} persons trained successfully.".format (len (np.unique (ids))))
		print("\n[INFO] Quitting the program")
	def Draw_Rect(self,Image, face,color):
		x,y,w,h = face
		cv2.line(Image, (x, y), (int(x + (w/5)),y), color, 2)
		cv2.line(Image, (int(x+((w/5)*4)), y), (x+w, y), color, 2)
		cv2.line(Image, (x, y), (x,int(y+(h/5))), color, 2)
		cv2.line(Image, (x+w, y), (x+w, int(y+(h/5))), color, 2)
		cv2.line(Image, (x, int(y+(h/5*4))), (x, y+h), color, 2)
		cv2.line(Image, (x, int(y+h)), (x + int(w/5) ,y+h), color, 2)
		cv2.line(Image, (x+int((w/5)*4), y+h), (x + w, y + h), color, 2)
		cv2.line(Image, (x+w, int(y+(h/5*4))), (x+w, y+h), color, 2)
	def create_dataset(self,samples,cam,dataset_name):
		fig, axs = plt.subplots(10,5, figsize=(20,20), facecolor='w', edgecolor='k')
		fig.subplots_adjust(hspace = .5, wspace=.001)
		#Names = self.FileRead()                               
		#print(Names)
		self.path_exists(dataset_name)
		count = 0 # Variable for counting the number of captured face photos
		face_id = self.Add_User()
		print("\n[INFO] Creating a dataset for further training purposes...")
		print ( "\n[INFO] Initializing the camera, please look in the camera lens and wait ...")
		while (True):
			# Capture, decode and return the next frame of the video
			ret, image = cam.read()	
			#Convert to gray-scale image
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#gray = cv2.equalizeHist(gray)
			# Search for faces in the gray-scale image
			# faces is an array of coordinates of the rectangles where faces exists
			faces = self._Face_Cascade.detectMultiScale(gray,scaleFactor = 1.098, minNeighbors = 6, minSize = (50, 50))
			# check if there are only 1 face in the photo
			if (len(faces) > 1):
				print("\n[Warning] there are more than one face !!!")
				continue
			try :
				for _,face in enumerate(faces):
					# Images with face coordinates
					# For gray_chunck, the coordinates are used for further transformation
					x, y, w, h = face
					gray_chunk = gray[y-30: y + h + 30, x-30: x + w + 30]
					image_chunk = image[y: y + h, x: x + w]
					# Search for the right eye 
					Right_Eye = self._Right_Eye_Cascade.detectMultiScale (gray[y: y + int (h / 2), x: x + int (w / 2)],
						scaleFactor = 1.05, minNeighbors = 6, minSize = (10, 10))
					# check if there only one right eye
					if len(Right_Eye) > 1:
						print("\n[Warning] Right Eye is not detected !!!")
						raise Exception
					for _,eye1 in enumerate(Right_Eye):
						rx, ry, rw, rh = eye1
						# Search for the left eye
						Left_Eye = self._Left_Eye_Cascade.detectMultiScale (gray [y: y + int (h / 2), x + int (w / 2): x +w],
							scaleFactor = 1.05, minNeighbors = 6, minSize = (10, 10))
						# check if there only one left eye
						if len(Left_Eye) > 1:
							print("\n[Warning] Left Eye is not detected !!!")
							raise Exception
						for _,eye2 in enumerate(Left_Eye):
							lx, ly, lw, lh =  eye2
							# Calculation of the angle between the eyes
							eyeXdis = (lx + w / 2 + lw / 2) - (rx + rw / 2)
							eyeYdis = (ly + lh / 2) - (ry + rh / 2)
							angle_rad = np.arctan (eyeYdis / eyeXdis)
							# convert degree to rad
							angle_degree = angle_rad * 180 / np.pi
							print("[INFO] Rotation angle : {:.2f} degree".format(angle_degree))
							# draw rectangles
							self.Draw_Rect(image, face,[0,255,0])
							cv2.rectangle (image_chunk, (rx, ry), (rx + rw, ry + rh), (255,255,255), 2)
							cv2.rectangle (image_chunk, (lx + int (w / 2), ly), (lx + int (w / 2) + lw, ly + lh), (0,255, 255), 2)
							cv2.imshow('Video', image)	
							# Image rotation 
							# Find the center of the image
							image_center = tuple(np.array(gray_chunk.shape) / 2)
							rot_mat = cv2.getRotationMatrix2D(image_center, angle_degree, 1.0)
							rotated_image = cv2.warpAffine(gray_chunk, rot_mat, gray_chunk.shape, flags=cv2.INTER_LINEAR)
							print("\n[INFO] Adding image number {} to the dataset".format(count))
							# Save the correct inverted image
							cv2.imwrite("dataset/Person." + str(face_id) + '.' + str(count) + ".jpg " ,
								rotated_image)
							axs[int(count/5)][count%5].imshow(rotated_image,cmap='gray', vmin=0, vmax=255)
							axs[int(count/5)][count%5].set_title("Person." + str(face_id) + '.' + str(count) + ".jpg ", 
								fontdict={'fontsize': 15,'fontweight': 'medium'})
							axs[int(count/5)][count%5].axis('off')
							'''
							count += 1
							cv2.imwrite("dataset/Person." + str(face_id) + '.' + str(count) + ".jpg " ,
								image_chunk)
							#self.Draw_Rect(rotated_image, face)
							axs[int(count/5)][count%5].imshow(gray_chunk,cmap='gray', vmin=0, vmax=255)
							axs[int(count/5)][count%5].set_title("Person." + str(face_id) + '.' + str(count) + ".jpg ", 
								fontdict={'fontsize': 15,'fontweight': 'medium'})
							axs[int(count/5)][count%5].axis('off')
							'''
							#print("[{},{}]".format(int(count/5),count%5))
							count += 1
							#cv2.imshow('Rotated to save', rotated_image)
					
			except Exception as e:
				print(e)
				print("[Warning] Something went wrong!!!")
				continue
			if cv2.waitKey (1) & 0xff == 27: #To exit the program, press "Esc", wait 100 ms,
				break
			elif count >= samples: # taking pic_num photos
				break
		print("\n[INFO] Dataset has been successfully created for this person...")
		cam.release ()
		cv2.destroyAllWindows ()
		plt.show()
def Arg_Parse():
	Arg_Par = arg.ArgumentParser()
	Arg_Par.add_argument("-v", "--video",
					help = "path of the video or if not then webcam")
	Arg_Par.add_argument("-c", "--camera",
					help = "Id of the camera")
	arg_list = vars(Arg_Par.parse_args())
	return arg_list
if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Please Provide an argument !!!")
		sys.exit(0)
	Arg_list = Arg_Parse()
	face_cascade = './Haar_Cascades/haarcascade_frontalface_default.xml'
	right_eye_cascade = './Haar_Cascades/haarcascade_righteye_2splits.xml'
	left_eye_cascade = './Haar_Cascades/haarcascade_lefteye_2splits.xml'
	if not (os.path.isfile(face_cascade)):
		raise RuntimeError("%s: not found" % face_cascade)
	if not (os.path.isfile(right_eye_cascade)):
		raise RuntimeError("%s: not found" % right_eye_cascade)
	if not (os.path.isfile(left_eye_cascade)):
		raise RuntimeError("%s: not found" % left_eye_cascade)
	samples = 50
	dataset_name = 'dataset/'
	file_name = 'train.yaml'
	# variables for LBPH algorithm
	radius = 1
	neighbour = 8
	grid_x = 8
	grid_y = 8
	var = list([radius,neighbour,grid_x,grid_y])
	model = Train_Model(face_cascade,right_eye_cascade,left_eye_cascade,var)
	if Arg_list["video"] != None :
		video = cv2.VideoCapture(Arg_list["video"])
		#create a dataset for further model training
		model.create_dataset(samples,video,dataset_name)
		#Training the model
		model.train(dataset_name,file_name)
	if Arg_list["camera"] != None :
		camera = cv2.VideoCapture(Arg_list["camera"])
		camera.set(3, 640)
		camera.set(4, 480)





