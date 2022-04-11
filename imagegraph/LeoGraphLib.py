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




def loadDropboxFolder(folderURL="https://www.dropbox.com/s/qxdfncdakdzm9yu/BHRBuildingSamples.zip?dl=0"):
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
	imlist = [im['arrays'] for im in imCollection]
	plt.figure(figsize=(15,15))
	
	imwidth = np.ceil((np.max(xCoords) - np.min(xCoords))/ (0.15*len(xCoords))*imSizeFactor)
	imheight = np.ceil((np.max(yCoords) - np.min(yCoords))/ (0.15*len(xCoords))*imSizeFactor)
	imwidth = np.max([imwidth, 0])
	imheight = np.max([imheight, 0])
	print('plotting images...')
	#plt.scatter(xCoords, yCoords)
	for i in tqdm.tqdm(range(len(xCoords))):
		thisim = imlist[i]

		thisAspectRatio = thisim.shape[0]/thisim.shape[1]

		left = xCoords[i]
		right = left+imwidth
		bottom = yCoords[i]
		top = bottom+np.ceil(imheight*thisAspectRatio)

		plt.imshow(thisim, extent=[left, right, bottom, top])

	plt.axis('off')
	plt.xlim([np.min(xCoords-imwidth), np.max(xCoords+imwidth)])
	plt.ylim([np.min(yCoords-imwidth), np.max(yCoords+imwidth)])
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


def showAllImages(imL):
	print('This function has still to be implemented. Try swapping me for saveAsGif(imL??)')
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

def displayNearestNeighbors(imCollection1, imCollection2, distanceMatrix, queryList=[], imageheight=100):

	# first, choose 4 random images to be centres...
	L = distanceMatrix.shape[1]
	w = distanceMatrix.shape[0]
	w2 = min(4,w)
	nearest = np.zeros((4,L)).astype(int)

	centres = []
	for i in range(w2):
		if queryList:
			centres.append(queryList[i])
		else:
			idx = np.random.randint(w)
			centres.append(idx)

	for i in range(w2):
		nearest[i,:] =  np.argsort(distanceMatrix[centres[i],:])

	htmlString = """<style>
	.igfilename {
	font-family: monospace;
	color:white;
	background-color: rgba(0,0,0,0.3);
	text-align:center !important;
	}"""
	urlList1 = [imc['urls'] for imc in imCollection1]
	imNames1 = [imc['meta'] for imc in imCollection1]
	urlList2 = [imc['urls'] for imc in imCollection2]
	imNames2 = [imc['meta'] for imc in imCollection2]
	for i in range(w2):
		htmlString += ".nnmyimage"+str(i)+"{display:none;} #myimage"+str(i)+":hover ~ .nnmyimage"+str(i)+"{display:inline-block;} #myimage"+str(i)+":hover{opacity:0.8;}"
	htmlString += """.imagebox{margin-top: 5px; margin-bottom:5px; float:left;width:25%;height:""" + str(imageheight) + """px;background-size: contain; background-position: center; background-repeat: no-repeat;}
	</style>
	<div style=" height:""" + str(imageheight*5) + """px; width:100%; display:inline-block; position:relative;">
	"""
	for i in range(min(4,len(imCollection1))):
		k = centres[i]
		htmlString += "<div class='imagebox' id='myimage"+str(i)+"' style='background-image: url( " + '"' + urlList1[k] + '"' + ")' > <p class='igfilename'>" + imNames1[k] +  "</p> </div> "
	htmlString +="""
		<div style="float:left; clear:left; background-color: lightgray; width:100%; height:2px;"> </div>  """
	for i in range(min(4,len(imCollection1))):
		for j in range(min(16,len(imCollection2))):
			htmlString+= "<div class='imagebox nnmyimage"+str(i)+"' style='background-image: url( " + '"' + urlList2[nearest[i,j]] + '"' + ")' > <p class='igfilename'>" + imNames2[nearest[i,j]] +  "</p> </div> "
	htmlString += "</div>"
	display(HTML(htmlString))
	return 0


## MISC 


def comment(commentText=" "):
	print(commentText)
	return 0
		
def consolePrint(anyinput):
	print(anyinput)
	return 0








