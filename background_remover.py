from sys import argv,exit
import os
import cv2
from functools import partial
import numpy as np
import glob
from multiprocessing import Pool
from PIL import Image, ImageChops

def removeBackground(image_file, out_folder):
	img = cv2.imread(image_file)

	im = Image.fromarray(img)
	bg = Image.new(im.mode, im.size, (255, 255, 255))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		im = im.crop(bbox)
		img = np.array(im)

	mask = np.zeros(img.shape[:2], dtype = np.uint8)

	height, width, _ = img.shape
	rect = (5, 5,width - 5,height - 5)

	bgdmodel = np.zeros((1,65),np.float64)
	fgdmodel = np.zeros((1,65),np.float64)

	firstTime = True
	for x in xrange(10):
		cv2.grabCut(img,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT if firstTime else cv2.GC_INIT_WITH_MASK)
		mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask2)
        firstTime = False

    #Convert black background to white
	output[np.where((output == [0,0,0]).all(axis = 2))] =[255,255,255]
	cv2.imwrite(os.path.join(out_folder, os.path.basename(image_file)), output) 

def main(in_folder, out_folder):
	images = glob.glob('%s/*.jpg' % in_folder)
	p = Pool(32)
	p.map(partial(removeBackground, out_folder=out_folder), images)

if __name__ == '__main__':
	try:
		in_folder=argv[1]
		out_folder=argv[2]
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)
	except:
		print "Provide the following arguments"
		print "1. Input folder name"
		print "2. Output folder name"
		exit()
	main(in_folder, out_folder)