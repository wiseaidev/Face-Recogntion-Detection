import numpy as np
import cv2
# Import the socket library
import socket
import pickle
import struct
from Recognize import *
if __name__ == "__main__":
	skin_detect = Skin_Detect()
	size1 = (30,30)
	size2 = (80,110)
	scale_factor = 3
	Face_Detect = Face_Detector(skin_detect)
	face_cascade = './Haar_Cascades/haarcascade_frontalface_default.xml'
	file_name = 'train.yaml'
	if not (os.path.isfile(file_name)):
		raise RuntimeError("%s: not found" % file_name)
	if not (os.path.isfile(face_cascade)):
		raise RuntimeError("%s: not found" % face_cascade)
	# variables for LBPH algorithm
	radius = 1
	neighbour = 8
	grid_x = 8
	grid_y = 8
	var = list([radius,neighbour,grid_x,grid_y])
	model = Recognizer(face_cascade,file_name,var)
	try:
		# Create a socket object
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		print("\n[*] Socket successfully created")
	except socket.error as err:
		print("\n[*] Socket creation failed with error : ",err)

	HOST = "localhost"
	# Port for socket
	PORT = 4444
	# Bind to the port
	server_socket.bind((HOST, PORT))
	print("\n[*] Socket binded to : ",PORT)
	# Put the socket into listening mode
	server_socket.listen(5)
	print("\n[*] Socket is now listening") 	 
	CHUNK_SIZE = 4 * 1024	
	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] 
	while True:
		# Establish connection with client
		print('\n[*] Waiting for client...')
		client_socket, addr = server_socket.accept()
		# print the socket object : ip addr and port nb
		print('\n[*] Connected from ip: {} and port : {} '.format(addr[0],addr[1]))
		while True:
			data = b""
			# struct_size is 8 bytes
			struct_size = struct.calcsize("l")
			print("\n[*] Struct Size: ",struct_size)
			img_size= client_socket.recv(struct_size)
			# struct.unpack retrun a tuple 
			img_size = struct.unpack("l", img_size)[0]
			print("\n[*] Message Size : {}".format(img_size))
			while len(data) < img_size:
				data += client_socket.recv(CHUNK_SIZE)
				print("\n[*] Receiving ",len(data))
			frame_data = data[:img_size]
			data = data[img_size:]
			frame=pickle.loads(frame_data)
			frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
			predicted = model.predict(frame,Face_Detect,size1,size2)
			result, frame = cv2.imencode('.jpeg', predicted, encode_param)
			# Returns the bytes object of the serialized object.
			data = pickle.dumps(frame, 0)
			size = len(data)
			#print("\n[*] Sending a packet size of: ",size)
			client_socket.sendall(struct.pack("l",size) + data)
			print("\n[*] Image is sent successfully ")
			# wainting for recognized images
			#print('\n[*] Waiting for Server...')
			#time.sleep(2)

