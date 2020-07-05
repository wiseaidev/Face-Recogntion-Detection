import socket
import numpy as np
import cv2
import pickle
import struct
import time
import sys
import argparse as arg
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
	if Arg_list["video"] != None :
		video = cv2.VideoCapture(Arg_list["video"])
	if Arg_list["camera"] != None :
		video = cv2.VideoCapture(eval(Arg_list["camera"]))
		video.set(3, 640)
		video.set(4, 480)
	HOST = 'localhost' 
	TCP_IP = socket.gethostbyname(HOST)  # Domain name resolution
	TCP_PORT = 4444    	 
	CHUNK_SIZE = 4 * 1024	 
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	# socket for sending and receiving images
	Client_Socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	print("\n[*] Connecting to server @ ip = {} and port = {}".format(TCP_IP,TCP_PORT))
	Client_Socket.connect((TCP_IP, TCP_PORT))
	print("\n[*] Client connected successfully !")
	while True:
		# Capture, decode and return the next frame of the video
		ret, image = video.read()
		#image = cv2.resize(image, (0,0), fx=1/3, fy=1/3)	
		result, frame = cv2.imencode('.jpeg', image, encode_param)
		# Returns the bytes object of the serialized object.
		data = pickle.dumps(frame, 0)
		size = len(data)
		#print("\n[*] Sending a packet size of: ",size)
		Client_Socket.sendall(struct.pack("l",size) + data)
		print("\n[*] Image is sent successfully ")
		# wainting for recognized images
		#print('\n[*] Waiting for Server...')
		#time.sleep(2)
		data = b""
		# struct_size is 8 bytes
		struct_size = struct.calcsize("l")
		#print("\n[*] Struct Size: ",struct_size)
		img_size= Client_Socket.recv(struct_size)
		# struct.unpack retrun a tuple 
		img_size = struct.unpack("l", img_size)[0]
		#print("\n[*] Message Size : {}".format(img_size))
		while len(data) < img_size:
			data += Client_Socket.recv(CHUNK_SIZE)
			#print("\n[*] Receiving ",len(data))
		frame_data = data[:img_size]
		data = data[img_size:]
		frame=pickle.loads(frame_data)
		frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
		cv2.imshow('Video', frame)	
		if cv2.waitKey (1) & 0xff == 27: #To exit the program, press "Esc", wait 100 ms,
				break


