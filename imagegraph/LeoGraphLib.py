import numpy as np
from matplotlib import pyplot as plt
import tqdm, glob
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn import manifold
import urllib, json
from PIL import Image
from io import BytesIO
import base64
import ssl
import requests
from IPython.core.display import display, HTML
from skimage import io, transform, color
from scipy.spatial import distance
import tensorflow as tf
import os, tarfile, sys
#Probably not good that skimage.io is refered to in the same way as the io library (as in io.BytesIO)

### Some helper functions here


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

## From Tensorflow Tutorials, slightly adapted. Some neural network helpers.
def maybe_download_and_extract():
	"""Download and extract model tar file."""
	DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
	model_dir = './'
	dest_directory = model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		print('Downloading neural network weights..')
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	else:
		print('Neural network weights already downloaded, extracting...')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_graph():
	"""Creates a graph from saved GraphDef file and returns a saver."""
	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(os.path.join('./', 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')



def run_inference_on_array(array, sess):
	image_data = imToBytes(array)
	softmax_tensor = sess.graph.get_tensor_by_name('pool_3:0')
	predictions = sess.run(softmax_tensor,
						   {'DecodeJpeg/contents:0': image_data})
	predictions = np.squeeze(predictions)
	return predictions

###  INPUTS




def loadGDriveImages(impaths='*.jpg' ):
	from google.colab import drive
	drive.mount('/content/drive')
	print('reading images...')
	imfilelist = glob.glob('/content/drive/My Drive/'+impaths)
	imCollection = []
	for imloc in tqdm.tqdm(imfilelist):
		tmpim = io.imread(imloc)
		if len(tmpim.shape) != 3:
			tmpim = color.gray2rgb(tmpim)
		if tmpim.shape[2] > 3:
			tmpim = tmpim[:,:,:3]
		thisimage = {}
		thisimage['arrays'] = tmpim
		thisimage['urls'] = encodeImage(tmpim)
		thisimage['meta'] = imloc.split('/')[-1]
		imCollection.append(thisimage)
	return imCollection


def loadIIIFManifest(manifestURL, maxDownload=100):
	print('downloading images from IIIF manifest...')

	imCollection = []
	with urllib.request.urlopen(manifestURL, context=ssl._create_unverified_context()) as url:
		data = json.loads(url.read().decode())
		canvases = data['sequences'][0]['canvases']
		for canvas in tqdm.tqdm(canvases):
			imloc = canvas['images'][0]['resource']['@id'] 
			imMeta = str(data['label']) + ': ' + str(canvas['label']) 
			try:
				#This means please give me at most 256x256 - see https://iiif.io/api/image/2.1/#size
				if imloc[-4:] != '.jpg':
					imloc += "/full/!256,256/0/default.jpg"
				tmpim = io.imread(imloc)
				if len(tmpim.shape) != 3:
					tmpim = color.gray2rgb(tmpim)
				thisimage = {}
				thisimage['arrays'] = tmpim
				thisimage['urls'] = imloc
				thisimage['meta'] = imMeta
				imCollection.append(thisimage)
			except (urllib.error.URLError, ssl.SSLError) as e:
				print('SSL cert invalid, downloading anyway')
				if imloc[-4:] != '.jpg':
					imloc += "/full/!256,256/0/default.jpg"
				tmpim = insecureImRead(imloc)
				if len(tmpim.shape) != 3:
					tmpim = color.gray2rgb(tmpim)
				thisimage = {}
				thisimage['arrays'] = tmpim
				thisimage['urls'] = imloc
				thisimage['meta'] = imMeta
				imCollection.append(thisimage)
			except:
				print('Image not downloaded... ')
				print(imloc)
				imCollection = []
	return imCollection



def injectFloat(floatstring='1.0'):
	return(float(floatstring))

### IMAGE PROCESSING


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



def getNNEmbedding(imCollection):
	maybe_download_and_extract()
	imlist = [im['arrays'] for im in imCollection]
	predictions = []
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	create_graph()
	with tf.Session(config=config) as sess:
		for thisim in tqdm.tqdm(imlist):
			predictions.append(run_inference_on_array(thisim, sess))
	return predictions

#### DATA MANIPULATINO

def reduceDims(vecList):
	my_tsne = manifold.TSNE(n_components=2)
	vecArr = np.asarray(vecList)
	xy = my_tsne.fit_transform(vecList)
	return xy[:,0], xy[:,1]


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
	distances = distance.cdist(embeddings1, embeddings2)
	return distances


### OUTPUTS

def scatterPlot(x, y):
	print('plotting xy...')
	plt.figure()
	plt.scatter(x, y)
	return 0


def consolePrint(anyinput):
	print(anyinput)
	return 0


def viewSingleImage(imCollection, whichImage):
	imlist = [im['arrays'] for im in imCollection]
	if whichImage is None:
		whichImage = 0
	whichImage = int(np.round(float(whichImage)))
	tmpim = imlist[whichImage]
	plt.imshow(tmpim)
	return 0


def pixPlot(imCollection, xCoords, yCoords):
	imlist = [im['arrays'] for im in imCollection]
	plt.figure(figsize=(15,15))
	imwidth = round((np.max(xCoords) - np.min(xCoords))/ (0.15*len(xCoords)))
	imheight = round((np.max(yCoords) - np.min(yCoords))/ (0.15*len(xCoords)))
	imwidth = np.max([imwidth, 0])
	imheight = np.max([imheight, 0])
	print('plotting images...')
	#plt.scatter(xCoords, yCoords)
	for i in tqdm.tqdm(range(len(xCoords))):
		thisim = imlist[i]
		left = xCoords[i]
		right = left+imwidth
		bottom = yCoords[i]
		top = bottom+imheight

		plt.imshow(thisim, extent=[left, right, bottom, top])
	plt.xlim([np.min(xCoords), np.max(xCoords)])
	plt.ylim([np.min(yCoords), np.max(yCoords)])
	return 0


def displayNearestNeighbors(imCollection1, imCollection2, distanceMatrix):

	# first, choose 4 random images to be centres...
	L = distanceMatrix.shape[1]
	nearest = np.zeros((4,L)).astype(int)

	centres = []
	for i in range(4):
		idx = np.random.randint(L)
		centres.append(idx)

	for i in range(4):
		nearest[i,:] =  np.argsort(distanceMatrix[centres[i],:])

	htmlString = """<style>
	.igfilename {
	font-family: monospace;
	color:white;
	background-color: rgba(0,0,0,0.1);
	text-align:center !important;
	}"""
	urlList1 = [imc['urls'] for imc in imCollection1]
	imNames1 = [imc['meta'] for imc in imCollection1]
	urlList2 = [imc['urls'] for imc in imCollection2]
	imNames2 = [imc['meta'] for imc in imCollection2]
	for i in range(min(4,len(imCollection1))):
		htmlString += ".nnmyimage"+str(i)+"{display:none;} #myimage"+str(i)+":hover ~ .nnmyimage"+str(i)+"{display:inline-block;} #myimage"+str(i)+":hover{opacity:0.8;}"
	htmlString += """.imagebox{margin-top: 5px; margin-bottom:5px; float:left;width:25%;height:100px;background-size: contain; background-position: center; background-repeat: no-repeat;}
	</style>
	<div style="height:600px; width:100%; display:inline-block; position:relative;">
	"""
	for i in range(min(4,len(imCollection1))):
		k = centres[i]
		htmlString += "<div class='imagebox' id='myimage"+str(i)+"' style='background-image: url( " + '"' + urlList1[k] + '"' + ")' > <p class='igfilename'>" + imNames1[k] +  "</p> </div> "
	htmlString +="""
		<div style="float:left; clear:left; background-color: lightgray; width:100%; height:2px;"> </div>  """
	for i in range(min(4,len(imCollection1))):
		for j in range(min(16,len(imCollection2))):
			htmlString+= "<div class='imagebox nnmyimage"+str(i)+"' style='background-image: url( " + '"' + urlList2[nearest[i,j]] + '"' + ")' > </div> "
	htmlString += "</div>"
	display(HTML(htmlString))
	return 0


def comment(commentText=" "):
	print(commentText)
	return 0
