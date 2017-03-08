import cv2
import math
import matplotlib.pyplot as plt

class imageUtils(object):
	"""utility class for image processing """
	
	def __init__(self):
		pass

	# Final image size 32x32x3 to be used by the CNN
	im_x = 32
	im_y = 32
	im_z = 3

	def plot_images(self, images):
		""" Function to plot test images """
		for index in range(len(images)):
		    plt.imshow(images[index])
		plt.show()

	def pre_process_image(self, img, xmin=None, ymin=None, xmax=None, ymax=None):
		"""
			Function to pre process images before feeding them into the network
		"""
		
		#self.plot_images([img])
		# Get image shape
		img_shape = img.shape
		# Crop the top and bottom to remove unwanted features
		if xmin is not None:
			img = img[ymin:ymax, xmin:xmax]
		img = cv2.resize(img,(self.im_x,self.im_y), interpolation=cv2.INTER_AREA)
		#self.plot_images([img])
		return img
