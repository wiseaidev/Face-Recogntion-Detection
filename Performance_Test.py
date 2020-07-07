import socket
import time
import cv2
import pickle
import struct
from threading import Thread
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 4444))
video = cv2.VideoCapture("videos/video2.mkv")
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
n = 0
ret, image = video.read()
result, frame = cv2.imencode('.jpeg', image, encode_param)
data = pickle.dumps(frame, 0)
size = len(data)
def monitor():
	global n
	while True:
		time.sleep(2)
		print(n, 'reqs/sec')
		n = 0
Thread(target=monitor).start()

while True:
	start = time.time()
	#print("\n[*] Sending a packet size of: ",size)
	sock.sendall(struct.pack("l",size) + data)
	resp =sock.recv(40000)
	n += 20
