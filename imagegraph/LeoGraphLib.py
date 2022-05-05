import numpy as np
from matplotlib import pyplot as plt
import tqdm, glob
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn import manifold, decomposition
import urllib, json
from PIL import Image
from io import BytesIO
import base64
import ssl
import requests
from IPython.core.display import display, HTML
from skimage import io, transform, color
from scipy.spatial import distance


import os, tarfile, sys, shutil
import skimage.feature
#Probably not good that skimage.io is refered to in the same way as the io library (as in io.BytesIO)

import pandas as pd

from bokeh.plotting import figure, output_file, show
from bokeh.io import show, output_notebook
from bokeh.layouts import layout, row, column
from bokeh.models import glyphs, ColumnDataSource, tools, CustomJS, Select, HoverTool, Label, LabelSet
from bokeh.models.widgets import Div




## Temporary stuff
def loadGNM(nImages='500'):
	nImages = int(nImages)
	import csv
	mycsv = []
	from google.colab import drive
	drive.mount('/content/drive', force_remount=True)
	with open("/content/drive/My Drive/Bilder GNM/gnm_data_red.csv2", "r") as myfile:
		csvreader = csv.reader(myfile)
		for row in csvreader:
			mycsv.append(row)
	print('Shuffling image database...')
	#np.random.shuffle(mycsv) # shuffle the image database
	imageList = []
	for row in tqdm.tqdm(mycsv[:nImages]):    
		try:
			thisimage = io.imread(row[0].split(';')[0])
			if len(thisimage.shape) != 3:
				thisimage = color.gray2rgb(thisimage)
			imDict = {}
			imDict['arrays'] = thisimage
			imDict['meta'] = row[1]
			imDict['urls'] = row[0].split(';')[0]
			imageList.append(imDict)
		except:
			print('I couldnt read image: ' + str(row[0]))
	return imageList





### Some helper functions here

def getSmaller(img, coords=False, maxsize = 256):
	width = img.shape[0]
	height = img.shape[1]
	if width>height:
		newwidth = maxsize
		newheight = int(maxsize*(height/width))
	else:
		newwidth = maxsize
		newheight = int(maxsize*(height/width))
	if coords:
		return [newwidth, newheight]
	else:
		img = transform.resize(img, (newwidth, newheight))
		if img.dtype == 'float64':
			img = (255*img).astype(np.uint8)
		return img



def encodeImage(myimage):
	pil_img = Image.fromarray(myimage)
	pil_img.thumbnail((128,128))
	buff = BytesIO()
	pil_img.save(buff, format="JPEG")
	new_image_string = "data:image/jpeg;base64," + base64.b64encode(buff.getvalue()).decode("utf-8")
	return new_image_string

def insecureImRead(imurl):
	response = requests.get(imurl, verify=False)
	img = Image.open(BytesIO(response.content))
	pix = np.array(img)
	return pix

def imToBytes(myimage):
	pil_img = Image.fromarray(myimage)
	pil_img.thumbnail((128,128))
	buff = BytesIO()
	pil_img.save(buff, format="JPEG")
	return buff.getvalue()



###  INPUTS



def loadIIIFManifest(manifestURL, maxDownload=1000):
	print('downloading images from IIIF manifest...')

	imCollection = []
	with urllib.request.urlopen(manifestURL, context=ssl._create_unverified_context()) as url:
		data = json.loads(url.read().decode())
		canvases = data['sequences'][0]['canvases']
		maxL = min(maxDownload, len(canvases)-1)
		for canvas in tqdm.tqdm(canvases[:maxL]):
			imloc = canvas['images'][0]['resource']['@id']
			imMeta = str(data['label']) + ': ' + str(canvas['label'])
			try:
				#This means please give me at most 256x256 - see https://iiif.io/api/image/2.1/#size
				if imloc[-4:] != '.jpg':
					imloc += "/full/!256,256/0/default.jpg"
				imloc.replace('/full/full/','/full/!256,256/')
				tmpim = io.imread(imloc)
				if len(tmpim.shape) != 3:
					tmpim = color.gray2rgb(tmpim)
				thisimage = {}
				thisimage['arrays'] = getSmaller(tmpim)
				thisimage['urls'] = imloc
				thisimage['meta'] = imMeta
				imCollection.append(thisimage)
			except (urllib.error.URLError, ssl.SSLError) as e:
				print('SSL cert invalid, downloading anyway')
				if imloc[-4:] != '.jpg':
					imloc += "/full/!256,256/0/default.jpg"
				imloc.replace('/full/full/','/full/!256,256/')
				tmpim = insecureImRead(imloc)
				if len(tmpim.shape) != 3:
					tmpim = color.gray2rgb(tmpim)
				thisimage = {}
				thisimage['arrays'] = getSmaller(tmpim)
				thisimage['urls'] = imloc
				thisimage['meta'] = imMeta
				imCollection.append(thisimage)
			except:
				print('Image not downloaded: ' + imloc)
				# print(imloc)
	return imCollection

def loadLocalImages(impaths='*.jpg', googleDrive=True):
	
	acceptedFileFormats = ['jpg','png','gif','jpeg','JPG','PNG','GIF','JPEG']
	
	if googleDrive:
		try:
			from google.colab import drive
			drive.mount('/content/drive')
			print('Google Colab env detected. Reading images from Google Drive...')
			imfilelist = []
			if '.' not in impaths:
				separator = '/*.'
				if impaths[-1] == '/':
					separator='*.'
				for extension in acceptedFileFormats:
					imfilelist += glob.glob('/content/drive/My Drive/' + impaths + separator + extension, recursive=True)
			else:
				imfilelist = glob.glob('/content/drive/My Drive/' + impaths, recursive=True)

		except:
			print('Not on Google Colab. Reading images from local drive...')
			imfilelist = []
			if '.' not in impaths:
				separator = '*/.'
				if impaths[-1] == '/':
					separator='*.'
				for extension in acceptedFileFormats:
					imfilelist += glob.glob( impaths + separator + extension, recursive=True)
			else:
				imfilelist = glob.glob('/content/drive/My Drive/' + impaths, recursive=True)
	else:
		print('Reading images from local drive...')
		imfilelist = []
		if '.' not in impaths:
			separator = '*/.'
			if impaths[-1] == '/':
				separator='*.'
			for extension in acceptedFileFormats:
				imfilelist += glob.glob( impaths + separator + extension, recursive=True)
		else:
			imfilelist = glob.glob(impaths, recursive=True)

	imCollection = []
	for imloc in tqdm.tqdm(imfilelist):
		if os.path.getsize(imloc) < 10000000:
			fileExten = imloc.split('.')[-1] 
			if fileExten in acceptedFileFormats:
				try:
					tmpim = io.imread(imloc)
					if len(tmpim.shape) != 3:
						tmpim = color.gray2rgb(tmpim)
					if tmpim.shape[2] > 3:
						tmpim = tmpim[:,:,:3]
					thisimage = {}
					thisimage['arrays'] = getSmaller(tmpim)
					thisimage['urls'] = encodeImage(tmpim)
					thisimage['meta'] = imloc.split('/')[-1]
					imCollection.append(thisimage)
				except:
					print('failed to read file: ' + imloc)
	return imCollection





def injectFloat(floatstring='1.0'):
	return(float(floatstring))




def loadDropboxFolder(folderURL="https://www.dropbox.com/sh/p5fngaj2mpccyjl/AACN0ErBKE1PgDh3LUrpJL4Ea?dl=0"):
# Written Leo Impett + Eva Cetinic 2022

  if (folderURL.split('?')[-1]=='dl=0'):
    print("Downloading "+folderURL+" to file")
    folderURL=folderURL.split('dl=')[0]+'dl=1'
    r = requests.get(folderURL, allow_redirects=True)
    with open('images.zip', 'wb') as myf:
      myf.write(r.content)



    outdir = "./images"
    k=0
    while os.path.isdir(outdir):
      k+=1
      outdir = "./images"+str(k)
    print("Unzipping folder " + outdir)

    shutil.unpack_archive("images.zip", outdir)
    print("Reading files from "+outdir+'/**/')

    return loadLocalImages(outdir+'/**/*',googleDrive=False)

### IMAGE PROCESSING



def getEntropy(imCollection ):
	print('extracting entropy...')
	imlist = [im['arrays'] for im in imCollection]
	e_out = []
	for image in tqdm.tqdm(imlist):
		imgray = color.rgb2gray(image)
		entr_img = entropy(imgray, disk(10))
		e = np.mean(entr_img.flatten())
		e_out.append(e)
	return e_out


def getNNEmbedding(imCollection, modelType='vgg16'):
	from tensorflow.keras.preprocessing import image
	from tensorflow.keras.applications.vgg16 import preprocess_input
	print("Initiating model: "+modelType)
	if modelType=='vgg16':
		from tensorflow.keras.applications.vgg16 import VGG16
		model = VGG16(weights='imagenet', include_top=False)
	if modelType=='inceptionv3':
		from tensorflow.keras.applications.inception_v3 import InceptionV3
		model = InceptionV3(weights='imagenet', include_top=False)
	if modelType=='efficientnet':
		from tensorflow.keras.applications import EfficientNetB3
		model = EfficientNetB3(weights='imagenet',include_top=False)
	imlist = [im['arrays'] for im in imCollection]
	predictions = []
	for thisim in tqdm.tqdm(imlist):
		x = np.expand_dims(transform.resize(thisim,(224,224)), axis=0)
		x = preprocess_input(x)
		predictions.append(model.predict(x).flatten())
	return predictions

def matchORB(imageCollection1, imageCollection2):

	descriptor_extractor = skimage.feature.ORB(n_keypoints=90)

	L1 = len(imageCollection1)
	L2 = len(imageCollection2)

	distanceMatrix = np.zeros((L1,L2))

	for i in tqdm.tqdm(range(L1)):
		img1 = imageCollection1[i]['arrays']
		descriptor_extractor.detect_and_extract(color.rgb2gray(img1))
		keypoints1 = descriptor_extractor.keypoints
		descriptors1 = descriptor_extractor.descriptors
		for j in range(L2):
			img2 = imageCollection2[j]['arrays']
			descriptor_extractor.detect_and_extract(color.rgb2gray(img2))
			keypoints2 = descriptor_extractor.keypoints
			descriptors2 = descriptor_extractor.descriptors
			matches12 = skimage.feature.match_descriptors(descriptors1, descriptors2, cross_check=True, max_distance=0.25)
			distanceMatrix[i,j] = 1.0/float(0.5+len(matches12))
	return distanceMatrix

def getRGB(imCollection ):
	print('extracting rgb...')
	imlist = [im['arrays'] for im in imCollection]
	r_out = []
	g_out = []
	b_out = []
	for image in tqdm.tqdm(imlist):
		[r, g, b] = np.mean(image,axis=(0,1))
		r_out.append(r)
		g_out.append(g)
		b_out.append(b)
	return r_out, g_out, b_out

def getHSV(imCollection ):
	print('extracting rgb...')
	imlist = [im['arrays'] for im in imCollection]
	h_out = []
	s_out = []
	v_out = []
	for image in tqdm.tqdm(imlist):
		[h, s, v] = np.mean(color.rgb2hsv(image),axis=(0,1))
		h_out.append(h)
		s_out.append(s)
		v_out.append(v)
	return h_out, s_out, v_out


def detectFaces(imList):
	import dlib
	detector = dlib.get_frontal_face_detector()
	faceImages = []
	for thisIm in imList:
		try:
			testimage = thisIm['arrays']
			dets = detector(testimage, 1)
			for myrect in dets:
				crp = [myrect.left(), myrect.right(), myrect.top(), myrect.bottom()]
				outim = getSmaller(testimage[crp[2]:crp[3],crp[0]:crp[1],:])
				faceImages.append({'arrays':outim, 'urls':encodeImage(outim), 'meta':thisIm['meta']})
		except:
			print("{} failed to detect".format(thisIm['meta']))
	return faceImages


#### DATA MANIPULATION

def reduceDims(vecList, method="TSNE"):
	if method not in ["TSNE", "UMAP", "PCA"]:
		raise ValueError("The *method* argument of ig.reduceDims should be one of: TSNE, UMAP, or PCA")
	
	vecArr = np.asarray(vecList)
	if method == "TSNE":
		reducer = manifold.TSNE(n_components=2)
	if method == "UMAP":
		import umap
		reducer = umap.UMAP(min_dist=2, spread=5)
	if method == "PCA":
		reducer = decomposition.PCA(n_components=2)
	xy = reducer.fit_transform(vecList)
	return xy[:,0], xy[:,1]


def mergeLists(a, b):
	return a+b



def toList(vecList, indexX, indexY):
	vecList = np.asarray(vecList)
	if indexX is None:
		return vecList[:,indexY].tolist()
	elif indexY is None:
		return vecList[indexX,:].tolist()
	else:
		return vecList[indexX,indexY]


def distanceMatrix(embeddings1, embeddings2):
	embeddings1 = np.asarray(embeddings1)
	embeddings2 = np.asarray(embeddings2)
	#np.reshape(embeddings1, (len(embeddings1), np.newaxis() ))
	distances = distance.cdist(embeddings1, embeddings2)
	return distances


def trainClassifier(x1, x2):
	from sklearn import svm
	x1 = np.asarray(x1)
	x2 = np.asarray(x2)

	X = np.concatenate((x1,x2),axis=0)

	y1 =  list(np.zeros(x1.shape[0]) )
	y2 =  list(np.ones(x2.shape[0]) )

	y = np.asarray(y1 + y2)

	myclassifier = svm.SVC(gamma='auto')
	myclassifier.fit(X, y)

	return myclassifier

def applyClassifier(imageList,vectorList,myclassifier):
	predictions = myclassifier.predict( np.asarray(vectorList) )
	scores = myclassifier.decision_function(  np.asarray(vectorList)  )
	myorder = list(np.argsort(scores))
	try:
		thresh = np.where(scores>0)[0][0]
		if thresh == 0:
			print('Everything is in class two!')
		sortOne = myorder[:thresh]
		k = myorder[thresh:]
		k.reverse()	
		sortTwo = k

	except:
		print('Everything is in class one! ')
		sortOne = myorder
		sortTwo = []

	imLOne = [imageList[i] for i in sortOne]
	imLTwo = [imageList[i] for i in sortTwo]

	return imLOne, imLTwo


### OUTPUTS

# Scatterplot with Plotly
def scatterPlot(X_in, Y_in):
	import plotly.express as px
	idx = ["#{}".format(x) for x in range(len(X_in))]
	fig = px.scatter(x=X_in, y=Y_in, hover_name=idx)
	fig.show()
	return 0

# Old Scatterplot
# def scatterPlot(x, y):
# 	print('plotting xy...')
# 	plt.figure()
# 	plt.scatter(x, y)
# 	return 0


def scatterImages(imCollection, xCoords, yCoords, imSizeFactor=1.0):
	output_notebook()
	
	imwidth = np.ceil((np.max(xCoords) - np.min(xCoords))/ (0.15*len(xCoords))*imSizeFactor)
	imheight = np.ceil((np.max(yCoords) - np.min(yCoords))/ (0.15*len(xCoords))*imSizeFactor)
	imwidth = np.max([imwidth, 0])
	imheight = np.max([imheight, 0])
	FrameRatio=(np.max(xCoords+imwidth)-np.min(xCoords-imwidth))/(np.max(yCoords+imwidth)-np.min(yCoords-imwidth))
	p = figure(aspect_scale=FrameRatio, match_aspect=True, x_range=[np.min(xCoords-imwidth), np.max(xCoords+imwidth)],y_range=[np.min(yCoords-imwidth), np.max(yCoords+imwidth)], active_scroll ="wheel_zoom", plot_width=800)
	print('plotting images...')
	for i in tqdm.tqdm(range(len(xCoords))):
		imAspectRatio=imCollection[i]['arrays'].shape[0]/imCollection[i]['arrays'].shape[1]
		if imAspectRatio <= 1.0:
			p.image_url(url=[imCollection[i]['urls']], x=xCoords[i], y=yCoords[i],w=imwidth,h=imwidth*imAspectRatio)
		else:
			p.image_url(url=[imCollection[i]['urls']], x=xCoords[i], y=yCoords[i],w=imwidth/imAspectRatio,h=imheight)
	
	p.xgrid.grid_line_color = None
	p.ygrid.grid_line_color = None
	p.xaxis.visible = False
	p.yaxis.visible = False
	show(p)
	return 0

def cart2pol(x, y):
	rho = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return(rho, phi)

def pol2cart(rho, phi):
	x = rho * np.cos(phi)
	y = rho * np.sin(phi)
	return(x, y)

def radialScatterImages(imCollection, angCoords, radCoords, axisRadius=750):
	imlist = [im['arrays'] for im in imCollection]
	plt.figure(figsize=(15,15))
	imwidth = 128
	imheight = 128
	imwidth = np.max([imwidth, 0])
	imheight = np.max([imheight, 0])
	print('plotting images...')
	xCoords = []
	yCoords = []
	meanAng = np.mean(angCoords)
	angCoords = [np.pi*ang/meanAng for ang in angCoords]
	meanRad = np.mean(radCoords)
	radCoords = [axisRadius*rad/meanRad for rad in radCoords]
	for i in range(len(angCoords)):
		[xcoord, ycoord] = pol2cart(radCoords[i],angCoords[i]);
		xCoords.append(xcoord)
		yCoords.append(ycoord)
		thisim = imlist[i]
		[newheight, newwidth] = getSmaller(thisim, coords=True, maxsize = 128)
		left = xcoord
		right = left+newwidth
		bottom = ycoord
		top = bottom+newheight

		plt.imshow(thisim, extent=[left, right, bottom, top])
	plt.axis('off')
	plt.xlim([np.min(xCoords-imwidth), np.max(xCoords+imwidth)])
	plt.ylim([np.min(yCoords-imwidth), np.max(yCoords+imwidth)])
	return 0


def saveAsGIF(imL, imsize=512):
	if imL:
		imfilePre = ''
		try:
			from google.colab import drive
			drive.mount('/content/drive')
			print('Google Colab env detected. Saving images to Google Drive...')
			imfilePre =  '/content/drive/My Drive/' 

		except:
			print('Saving to local drive...')
			
		import imageio
		metastring = str(imL[0]['meta'])
		filestring = imfilePre + "".join([c for c in metastring if c.isalnum()]) + str(np.random.randint(999)) + '.gif'
		print(filestring)
		imageio.mimsave(filestring, [np.zeros((imsize, imsize, 3)).astype(np.uint8)] + [getSmaller(im['arrays'], maxsize=imsize) for im in imL], format='GIF', duration=0.5)
	return 0


def showAllImages(imCollection):
	data=np.array([np.squeeze(np.expand_dims(transform.resize(im['arrays'],(224,224)), axis=0), axis=0) for im in imCollection])
	if len(data.shape) == 3:
		data = np.tile(data[...,np.newaxis], (1,1,1,3))
	data = data.astype(np.float32)
	min = np.min(data.reshape((data.shape[0], -1)), axis=1)
	data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
	max = np.max(data.reshape((data.shape[0], -1)), axis=1)
	data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
	n = int(np.ceil(np.sqrt(data.shape[0])))
	padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
	data = np.pad(data, padding, mode='constant', constant_values=0)
	data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
	data = (data * 255).astype(np.uint8)
	plt.figure(figsize = (12,12))
	plt.axis('off')
	plt.imshow(data)
	return 0


def viewSingleImage(imCollection, whichImage):
	imlist = [im['arrays'] for im in imCollection]
	if whichImage is None:
		whichImage = 0
	whichImage = int(np.round(float(whichImage)))
	tmpim = imlist[whichImage]
	plt.imshow(tmpim)
	return 0

def showTwoImageSets(imCollection1, imCollection2):
	plt.figure(figsize=(15,15))

	for i in range(9):
		if len(imCollection1) > i:
			plt.subplot(9,2,(i*2)+1)
			plt.imshow(imCollection1[i]['arrays'])

		if len(imCollection2) > i:
			plt.subplot(9,2,(i*2)+2)
			plt.imshow(imCollection2[i]['arrays'])

	return 0

def displayNearestNeighbors(imCollection1, imCollection2, distanceMatrix):
	
	output_notebook()
	
	imgs1 = [imCollection1[x]['urls'] for x in range(len(imCollection1))]
	meta1 = [imCollection1[x]['meta'] for x in range(len(imCollection1))]
	imAspectRatio=[imCollection1[x]['arrays'].shape[0]/imCollection1[x]['arrays'].shape[1] for x in range(len(imCollection1))]
	imgs2 = [imCollection2[x]['urls'] for x in range(len(imCollection2))]
	meta2=[imCollection2[x]['meta'] for x in range(len(imCollection2))]
	NNind=[]
	NNurls=[]
	NNmetas=[]
	for i in range(len(imCollection1)):
		NNind.append(list(np.argsort(distanceMatrix[i])[1:11]))
		NNurl=[]
		NNmeta=[]
		for j in list(np.argsort(distanceMatrix[i])[1:11]):
			NNurl.append(imgs2[j])
			NNmeta.append(meta2[j])
		NNurls.append(NNurl)
		NNmetas.append(NNmeta)

	df=pd.DataFrame({})
	df['url']=imgs1
	df['meta']=meta1
	df['ar']=imAspectRatio
	df['10nn'] = NNind
	df['10nn_urls'] = NNurls
	df['10nn_meta'] = NNmetas
	
	source = ColumnDataSource(df)
	source2 = ColumnDataSource(data=dict(
	x=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
	y=10*[2],
	color=['#990000','#b30000','#e60000','#ff0000','#ff3333','#ff4d4d','#ff8080','#ffb3b3', '#ffcccc','#ffe6e6'],
	width=10*[1], 
	height=10*[0.6],
	NN=df['10nn_meta'][0]))
	
	source3=ColumnDataSource(data=dict(
		url=[df['url'].tolist()[0]],
		x=[0],
		y=[1],
		w=[1],
		h=[df['ar'].tolist()[0]]
	))

	p = figure(width=200, match_aspect=True, aspect_ratio=1,toolbar_location=None)
	im = p.image_url(source=source3, x='x', y='x', w='w', h='h')
	p.ygrid.grid_line_color = None
	p.xaxis.visible = False
	p.yaxis.visible = False
	p.xgrid.grid_line_color = None
	p.outline_line_color= None
	
	p2= figure(x_range=(0,10), y_range=(0,1), match_aspect=True, aspect_ratio=10, width=900, toolbar_location=None )
	im1=p2.image_url(url=[source.data['10nn_urls'][0][0]], x=0, y=1, w=1, h=1)
	im2=p2.image_url(url=[source.data['10nn_urls'][0][1]], x=1, y=1, w=1, h=1)
	im3=p2.image_url(url=[source.data['10nn_urls'][0][2]], x=2, y=1, w=1, h=1)
	im4=p2.image_url(url=[source.data['10nn_urls'][0][3]], x=3, y=1, w=1, h=1)
	im5=p2.image_url(url=[source.data['10nn_urls'][0][4]], x=4, y=1, w=1, h=1)
	im6=p2.image_url(url=[source.data['10nn_urls'][0][5]], x=5, y=1, w=1, h=1)
	im7=p2.image_url(url=[source.data['10nn_urls'][0][6]], x=6, y=1, w=1, h=1)
	im8=p2.image_url(url=[source.data['10nn_urls'][0][7]], x=7, y=1, w=1, h=1)
	im9=p2.image_url(url=[source.data['10nn_urls'][0][8]], x=8, y=1, w=1, h=1)
	im10=p2.image_url(url=[source.data['10nn_urls'][0][9]], x=9, y=1, w=1, h=1)
	p2.ygrid.grid_line_color = None
	p2.xaxis.visible = False
	p2.yaxis.visible = False
	p2.xgrid.grid_line_color = None
	
	hover = HoverTool(tooltips=[('', '@NN')])
	p3 = figure(x_range=(0,10), y_range=(0,2), match_aspect=True, aspect_ratio=10, width=900, toolbar_location=None )
	p3.rect('x','y',color='color', source=source2, width='width', height='height')
	p3.add_tools(hover)
	p3.ygrid.grid_line_color = None
	p3.xaxis.visible = False
	p3.yaxis.visible = False
	p3.xgrid.grid_line_color = None
	p3.outline_line_color= None

	cb = CustomJS(args=dict(im=im, im1=im1, source=source, source2=source2, source3=source3, im2=im2, im3=im3, im4=im4, im5=im5, im6=im6, im7=im7, im8=im8, im9=im9, im10=im10 ), code="""
		  for (var i = 0; i <= source.data['url'].length; i++){
			  if (source.data['meta'][i] == cb_obj.value) {
				  var b=i;
			  } 
		  }
		  var data3 = source3.data;
		  var url= data3['url'];
		  var h = data3['h'];
		  h[0] = source.data['ar'][b];
		  url[0] = source.data['url'][b];
		  source3.change.emit();

		  im1.data_source.data['url'] = [source.data['10nn_urls'][b][0]];
		  im1.data_source.change.emit();
		  im2.data_source.data['url'] = [source.data['10nn_urls'][b][1]];
		  im2.data_source.change.emit();
		  im3.data_source.data['url'] = [source.data['10nn_urls'][b][2]];
		  im3.data_source.change.emit();
		  im4.data_source.data['url'] = [source.data['10nn_urls'][b][3]];
		  im4.data_source.change.emit();
		  im5.data_source.data['url'] = [source.data['10nn_urls'][b][4]];
		  im5.data_source.change.emit();
		  im6.data_source.data['url'] = [source.data['10nn_urls'][b][5]];
		  im6.data_source.change.emit();
		  im7.data_source.data['url'] = [source.data['10nn_urls'][b][6]];
		  im7.data_source.change.emit();
		  im8.data_source.data['url'] = [source.data['10nn_urls'][b][7]];
		  im8.data_source.change.emit();
		  im9.data_source.data['url'] = [source.data['10nn_urls'][b][8]];
		  im9.data_source.change.emit();
		  im10.data_source.data['url'] = [source.data['10nn_urls'][b][9]];
		  im10.data_source.change.emit();

		  const data2 = source2.data;
		  const x = data2['x'];
		  const y = data2['y'];
		  const nn = data2['NN'];
		  for (let i = 0; i < x.length; i++) {
				  nn[i] = source.data['10nn_meta'][b][i]
			  }
		  source2.change.emit();

	   """)	
	
	div = Div(text="Ten most similar images")
	options=df['meta'].tolist()
	select = Select(title="Image file name:", value=df['meta'].tolist()[0], options=options, width=190)
	select.js_on_change("value", cb)
	layout=row(column(select,p), column(div, p2, p3) )
	show(layout)
	return 0


## MISC 


def comment(commentText=" "):
	print(commentText)
	return 0
		
def consolePrint(anyinput):
	print(anyinput)
	return 0








