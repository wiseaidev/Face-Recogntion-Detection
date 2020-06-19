import math as m
import cv2 
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
class Visualize_LBP():
	def __init__(self,Radius=1,neighbors=4):
		self._radius = Radius
		self._neighbors = neighbors
	@property
	def radius(self):
		return self._radius
	@property
	def neighbors(self):
		return self._neighbors
	def compute_LBP(self,img):
		img1 = img.copy()
		for i in range(m.ceil(self._radius),img.shape[0] - m.ceil(self._radius)):
			for j in range(m.ceil(self._radius),img.shape[1] - m.ceil(self._radius)):
				window = img[i - m.ceil(self._radius) : i + m.ceil(self._radius) + 1,
				j - m.ceil(self._radius) : j + m.ceil(self._radius) + 1]
				window_lbp =  self.LBP(window)
				img[i,j] = window_lbp
				cv2.imshow('faces', img)
				#cv2.imshow('faces', img1)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					sys.exit(0)
	def LBP(self,img_chunk):
		#coordinates, using ceil because the block is bigger than circle
		gc_x = m.ceil(self._radius)
		gc_y = gc_x
		#pixel value
		gc = img_chunk[gc_x, gc_y]
		print("gc=====",gc)
		gp_xs = []
		gp_ys = []
		pixels = []
		angles = []
		lbp_result = 0
		for p in range(self._neighbors):
			theta = 2 * m.pi * p /self._neighbors
			angles.append(theta)
			gp_x = - self._radius * m.sin(theta)
			gp_x = round(gp_x,4)#get 4 decimals
			gp_xs.append(gp_x)
			gp_y = + self._radius * m.cos(theta)
			gp_y = round(gp_y,4)#get 4 decimals
			gp_ys.append(gp_y)
			#get the fraction part of gp_x
			gp_x_fract = gp_x - m.floor(gp_x)
			#get the fraction part of gp_y
			gp_y_fract = gp_y - m.floor(gp_y)
			gp_x_trans = round(gp_x + m.ceil(self._radius),4)
			gp_y_trans = round(gp_y + m.ceil(self._radius),4)
			#if the fraction parts are zeros, then no need to interpolate
			if gp_x_fract == 0 and gp_y_fract == 0:
				pixel = img_chunk[int(gp_x_trans),int(gp_y_trans)]
				pixels.append(pixel)
			else:
				points = []
				if 0 <= theta <=  m.pi/2:
					q11 = gc
					points.append(q11)
					q21 = img_chunk[gc_y , -1 ] 
					points.append(q21)
					q22 = img_chunk[0,gc_x * 2 ]
					points.append(q22)
					q12 = img_chunk[0,gc_x]
					points.append(q12)
					x1 = y1 = 0
					x2 = y2 = gc_x
					points.append(x1)
					points.append(y1)
					points.append(x2)
					points.append(y2)
					#[q11,q21,q22,q12,x1,y1,x2,y2]
					pixel = int(self.bilinear_interpolation(abs(gp_y),abs(gp_x),points))
				if m.pi/2 < theta <=  m.pi:
					q11 = img_chunk[gc_x , 0 ] 
					points.append(q11)
					q21 = gc
					points.append(q21)
					q22 = img_chunk[0,gc_x]
					points.append(q22)
					q12 = img_chunk[0,0]
					points.append(q12)
					x1 = y1 = 0
					x2 = y2 = gc_x
					points.append(x1)
					points.append(y1)
					points.append(x2)
					points.append(y2)
					#[q11,q21,q22,q12,x1,y1,x2,y2]
					pixel = int(self.bilinear_interpolation(abs(gp_x_trans),abs(gp_y_trans),points))
				if m.pi < theta <=  3*m.pi/2:
					q11 = img_chunk[-1 , 0] 
					points.append(q11)
					q21 = img_chunk[-1 , gc_x] 
					points.append(q21)
					q22 = gc
					points.append(q22)
					q12 = img_chunk[gc_x,0]
					points.append(q12)
					x1 = y1 = 0
					x2 = y2 = gc_x
					points.append(x1)
					points.append(y1)
					points.append(x2)
					points.append(y2)
					#[q11,q21,q22,q12,x1,y1,x2,y2]
					pixel = int(self.bilinear_interpolation(abs(gp_x_trans),abs(gp_x_trans),points))
				if 3*m.pi/2 < theta <=  2*m.pi:
					q11 = img_chunk[-1 , gc_x] 
					points.append(q11)
					q21 = img_chunk[-1 , -1] 
					points.append(q21)
					q22 = img_chunk[gc_x , -1] 
					points.append(q22)
					q12 = gc
					points.append(q12)
					x1 = y1 = 0
					x2 = y2 = gc_x
					points.append(x1)
					points.append(y1)
					points.append(x2)
					points.append(y2)
					#[q11,q21,q22,q12,x1,y1,x2,y2]
					pixel = int(self.bilinear_interpolation(gp_x,gp_y,points))
			lbp_result += self.pixel_compare(gc,pixel)*2**p
		return lbp_result
	def bilinear_interpolation(self,x,y,points):
		q11,q21,q22,q12,x1,y1,x2,y2 = points
		return (q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) + q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
	def pixel_compare(self,gc,pixel):
		return 1 if pixel >= gc else 0
class Fast_LBP():
	def __init__(self, radius=1, neighbors=8):
		self._radius = radius
		self._neighbors = neighbors
	@property
	def radius(self):
		return self._radius
	@property
	def neighbors(self):
		return self._neighbors		
	def Compute_LBP(self,Image):
		#Determine the dimensions of the input image.
		ysize, xsize = Image.shape
		# define circle of symetrical neighbor points
		angles_array = 2*np.pi/self._neighbors
		alpha = np.arange(0,2*np.pi,angles_array)
		# Determine the sample points on circle with radius R
		s_points = np.array([-np.sin(alpha), np.cos(alpha)]).transpose()
		s_points *= self._radius
		# s_points is a 2d array with 2 columns (y,x) coordinates for each cicle neighbor point		
		# Determine the boundaries of s_points wich gives us 2 points of coordinates
		# gp1(min_x,min_y) and gp2(max_x,max_y), the coordinate of the outer block 
		# that contains the circle points
		min_y=min(s_points[:,0])
		max_y=max(s_points[:,0])
		min_x=min(s_points[:,1])
		max_x=max(s_points[:,1])
		# Block size, each LBP code is computed within a block of size bsizey*bsizex
		# so if radius = 1 then block size equal to 3*3
		bsizey = np.ceil(max(max_y,0)) - np.floor(min(min_y,0)) + 1
		bsizex = np.ceil(max(max_x,0)) - np.floor(min(min_x,0)) + 1
		# Coordinates of origin (0,0) in the block
		origy =  int(0 - np.floor(min(min_y,0)))
		origx =  int(0 - np.floor(min(min_x,0)))
		#Minimum allowed size for the input image depends on the radius of the used LBP operator.
		if xsize < bsizex or ysize < bsizey :
			raise Exception('Too small input image. Should be at least (2*radius+1) x (2*radius+1)')
		# Calculate dx and dy: output image size
		# for exemple, if block size is 3*3 then we need to substract the first row and the last row which is 2 rows
		# so we need to substract 2, same analogy applied to columns
		dx = int(xsize - bsizex + 1)
		dy = int(ysize - bsizey + 1)
		# Fill the center pixel matrix C.
		C = Image[origy:origy+dy,origx:origx+dx]
		# Initialize the result matrix with zeros.
		result = np.zeros((dy,dx), dtype=np.float32)
		for i in range(s_points.shape[0]):
			# Get coordinate in the block:
			p = s_points[i][:]
			y,x = p + (origy, origx)
			# Calculate floors, ceils and rounds for the x and ysize
			fx = int(np.floor(x))
			fy = int(np.floor(y))
			cx = int(np.ceil(x))
			cy = int(np.ceil(y))
			rx = int(np.round(x))
			ry = int(np.round(y))
			D = [[]]
			if np.abs(x - rx) < 1e-6 and np.abs(y - ry) < 1e-6:
			#Interpolation is not needed, use original datatypes
				N = Image[ry:ry+dy,rx:rx+dx]
				D = (N >= C).astype(np.uint8)
			else:
				# interpolation is needed
				# compute the fractional part.
				ty = y - fy
				tx = x - fx
				# compute the interpolation weight.
				w1 = (1 - tx) * (1 - ty)
				w2 = tx * (1 - ty)
				w3 = (1 - tx) * ty
				w4 = tx * ty
				# compute interpolated image:
				N = w1*Image[fy:fy+dy,fx:fx+dx]
				N = np.add(N, w2*Image[fy:fy+dy,cx:cx+dx], casting="unsafe")
				N = np.add(N, w3*Image[cy:cy+dy,fx:fx+dx], casting="unsafe")
				N = np.add(N, w4*Image[cy:cy+dy,cx:cx+dx], casting="unsafe")
				D = (N >= C).astype(np.uint8)
			#Update the result matrix.
			v = 2**i
			result += D*v
			cv2.imshow('faces', result)
			if cv2.waitKey(800) & 0xFF == ord("q"):
				pass
		return result.astype(np.uint8)

class LBP_sklearn:
	def __init__(self, Nb_Points, Radius):
		# Initiate number of points(neighbors) and the radius of the cercle
		self._Nb_Points = Nb_Points
		self._Radius = Radius
	@property
	def Radius(self):
		return self._Radius
	@property
	def Nb_Points(self):
		return self._Nb_Points
	def compute(self, gray):
		# compute the Local Binary Pattern of the image, 
		# and then use the LBP representation
		# to build the histogram of patterns
		LBP = local_binary_pattern(gray, self._Nb_Points,self._Radius, method="uniform")
		axs[1][1].imshow(LBP,cmap='gray', vmin=0, vmax=9)
		axs[1][1].set_title('LBP Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
		axs[1][1].axis('off')
		(hist, bins) = np.histogram(LBP.ravel(),
		bins=np.arange(0, self._Nb_Points + 3),
		range=(0,self._Nb_Points + 2))
		width = bins[1] - bins[0]
		center = (bins[:-1] + bins[1:]) / 2
		axs[1][0].bar(center, hist, align='center', width=width)
		axs[1][0].set_title('Histogram', fontdict={'fontsize': 15, 'fontweight': 'medium'})
		# normalize the histogram
		hist = hist.astype("float")
		hist /=hist.sum() 
		# return the histogram of Local Binary Patterns
		return hist




if __name__ == "__main__":
	var_lbp = Visualize_LBP(Radius = 1,neighbors = 8)
	matrix1 = np.array( [[25, 41, 24],[29, 33, 80],[38, 56, 65]], dtype=np.int16)
	matrix2 = np.array( [[50, 60, 70, 80, 90],[51, 61, 2, 81, 3],[52, 62, 72, 82, 92],[53, 13, 73, 43, 93],[54, 64, 74, 22, 1]], dtype=np.uint8)
	#print(matrix2)
	lbp2 = Fast_LBP(1,8)
	#result = lbp2.Compute_LBP(matrix2)
	#print("result2 = ",result)
	#var_lbp.compute_LBP(matrix2)

	#print("result2 = \n",result)

	img = np.array(cv2.imread("images/img1.png",cv2.IMREAD_GRAYSCALE), 'uint8')
	img1 = np.array(cv2.imread("images/img1.png"), 'uint8')
	cv2.imshow('faces', img)
	if cv2.waitKey(1000) & 0xFF == ord("q"):
		pass
	#result = var_lbp.compute_LBP(img)
	result1 = lbp2.Compute_LBP(img)
	#cv2.imshow('faces', result)
	cv2.imshow('faces', result1)
	if cv2.waitKey(0) & 0xFF == ord("q"):
		pass	

	# initialize the local binary patterns descriptor along with
	# the data and label lists
	lbp = LBP_sklearn(8, 1)

	fig, axs = plt.subplots(2,2, figsize=(80,80), facecolor='w', edgecolor='k')
	fig.subplots_adjust(hspace = .5, wspace=.001)

	gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
	gray = gray(img1)
	axs[0][0].imshow(img1,cmap='gray', vmin=0, vmax=255)
	axs[0][0].set_title('Original Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
	axs[0][0].axis('off')

	axs[0][1].imshow(gray,cmap='gray', vmin=0, vmax=255)
	axs[0][1].set_title('GrayScale Image', fontdict={'fontsize': 15, 'fontweight': 'medium'})
	axs[0][1].axis('off')
	hist = lbp.compute(gray)
	# extract the label from the image path, then update the

	plt.show()
