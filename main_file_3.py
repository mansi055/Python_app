from PIL import Image, ImageEnhance, ImageFilter
import pytesseract,cv2,numpy,time
import numpy as np
import argparse
import os
import  pillowfight,imutils
from imutils.object_detection import non_max_suppression

config = '-l eng --psm 7 --oem 1'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def convertCv2ImageToPilImage(image):
    cv2.imwrite("t.png",image)
    return Image.open("t.png")
def convertPilImageToCv2Image(image):
    image.save("t.png")
    return cv2.imread("t.png")
def copyCv2Image(image):
	cv2.imwrite("t.png", image)
	return cv2.imread("t.png")

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
# print(args["input"])
# load the input image from disk

# PATH_ORIGINAL = args["input"]
# PATH_ORIGINAL = 'C:/Users/Netan/Desktop/skew_test_2.jpg'
PATH_SWT = 'imgSwt.jpg'
PATH_CROP_AFTER_EAST = 'cropAfterEast.jpg'
PATH_ORIGINAL_ROTATED = 'rotatedOriginalImage.jpg'
PATH_300_DPI_IMAGE = "300dpiImage.jpg"


def east(image):
	imageCopied = copyCv2Image(image)
	(H, W) = imageCopied.shape[:2]
	(newW, newH) = (W-(W%32)+32,H-(H%32)+32)
	rW = W / float(newW)
	rH = H / float(newH)
	 
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	# print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet('frozen_east_text_detection.pb')
	 
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	 
	# show timing information on text prediction
	# print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	results=[]
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.95:
				continue
	 
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
	 
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
	 
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
	 
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
	 
			# add the bounding box coordinates and probability score to
			# our respective lists
			
			rects.append((startX-4, startY+4, endX-4, endY+4))
			confidences.append(scoresData[x])
			# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
	 
	# show the output image
	# cv2.imshow("Text Detection", image)
	# cv2.waitKey(0)
	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	 
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX-rW)
		startY = int(startY-rH)
		endX = int(endX)
		endY = int(endY)
	 
		# draw the bounding box on the image
		roi = image[startY-4:endY+4, startX:endX+4]
		# in order to apply Tesseract v4 to OCR text we must supply
		# (1) a language, (2) an OEM flag of 4, indicating that the we
		# wish to use the LSTM neural net model for OCR, and finally
		# (3) an OEM value, in this case, 7 which implies that we are
		# treating the ROI as a single line of text
		# config = ("-l eng  --psm 11")
		# text = pytesseract.image_to_string(roi, config=config)
		text = ''
		# add the bounding box coordinates and OCR'd text to the list
		# of results
		results.append(((startX, startY, endX, endY), text))
		# results.append(text)
	results = sorted(results, key=lambda r:r[0][1])
	return results, image

def main_func(image_path):
	#start
	PATH_ORIGINAL = image_path
	originalPath = PATH_ORIGINAL
	imgSkewed = Image.open(originalPath)
	imgSwt = pillowfight.swt(imgSkewed, output_type=pillowfight.SWT_OUTPUT_BW_TEXT)
	imgSwt.save(PATH_SWT)
	# saved skewed Image
	# imageCopied = convertPilImageToCv2Image(imgSwt)
	image = convertPilImageToCv2Image(imgSwt)

	results, image = east(image)

	# TODO
	rect = results[-4][0]
	# import pdb
	# pdb.set_trace()

	roi = image[rect[1]-40:rect[3]+40, rect[0]-40:rect[2]+40]
	if roi==[]:
		roi = image[rect[1]-4:rect[3], rect[0]-4:rect[2]]
	# cv2.imwrite(PATH_CROP_AFTER_EAST, roi)
	# roi = image[rect[1]-4:rect[3], rect[0]-4:rect[2]]
	image = roi

	# for rect in results[-2]:
		# roi = image[rect[1]-40:rect[3]+40, rect[0]-40:rect[2]+40]
		# cv2.imwrite('crop.jpg',roi)
		# break

	#################################################################################################

	# convert the image to grayscale and flip the foreground
	# and background to ensure foreground is now "white" and
	# the background is "black"
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	 
	# threshold the image, setting all foreground pixels to
	# 255 and all background pixels to 0
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# grab the (x, y) coordinates of all pixel values that
	# are greater than zero, then use these coordinates to
	# compute a rotated bounding box that contains all
	# coordinates
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	 
	# the `cv2.minAreaRect` function returns values in the
	# range [-90, 0); as the rectangle rotates clockwise the
	# returned angle trends to 0 -- in this special case we
	# need to add 90 degrees to the angle
	if angle < -45:
		angle = -(90 + angle)
	 
	# otherwise, just take the inverse of the angle to make
	# it positive
	else:
		angle = -angle

	# print("angle " + str(angle))

	# rotate the image to deskew it
	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	img = cv2.imread(PATH_ORIGINAL)
	(h, w) = img.shape[:2]
	rotated = cv2.warpAffine(img, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	# # draw the correction angle on the image so we can validate it
	# cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	 
	# show the output image
	# print("[INFO] angle: {:.3f}".format(angle))
	# cv2.imshow("Input", image)
	# cv2.imshow("Rotated", rotated)
	# cv2.imwrite(PATH_ORIGINAL_ROTATED,rotated)
	# cv2.waitKey(0)
	#################################################################
	image = convertCv2ImageToPilImage(rotated)
	image.save(PATH_300_DPI_IMAGE,dpi=(300,300))

	results, image = east(convertPilImageToCv2Image(image))

	# print(results)
	######################################################################################################33\

	imgSkewed = Image.open(PATH_300_DPI_IMAGE)
	# imgSwt = pillowfight.swt(imgSkewed, output_type=pillowfight.SWT_OUTPUT_BW_TEXT)
	imgSkewed.save(PATH_SWT)

	cmd = "tesseract " + PATH_SWT + " test -l eng --psm 11 --oem 1" 

	returned_value = os.system(cmd)
	  # returns the exit code in unix
	text = open("test.txt", "r")
	text1 = str(text.read())
	print(text.read())
	return (text1)

# print('returned value:', returned_value)

