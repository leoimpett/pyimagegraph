import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform, color
import tqdm, glob
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sklearn import manifold
import urllib, json
 
###  INPUTS  

def loadImages(impaths='/Users/impett/Documents/Code/DigitalArtHistory/Notebooks/images/*.jpg' ):
    print('reading images...')
    imfilelist = glob.glob(impaths)
    imlist = []
    for imloc in tqdm.tqdm(imfilelist):
        tmpim = io.imread(imloc)
        if len(tmpim.shape) != 3:
            tmpim = color.gray2rgb(tmpim)
        if tmpim.shape[2] > 3:
            tmpim = tmpim[:,:,:3]
        imlist.append(tmpim)
    return imlist


def loadIIIFManifest(manifestURL):
    print('downloading images from IIIF manifest...')
    with urllib.request.urlopen(manifestURL) as url:
        data = json.loads(url.read().decode())
    canvases = data['sequences'][0]['canvases']
    imurls = [canvas['images'][0]['resource']['@id'] for canvas in canvases]
    imlist = []
    for imloc in tqdm.tqdm(imurls):
        # Reduce the quality to 256
        myurl = imloc.split('/')
        #This means please give me at most 256x256 - see https://iiif.io/api/image/2.1/#size 
        myurl[-3] = '!256,256'
        imloc = "/".join(myurl)
        
        tmpim = io.imread(imloc)
        if len(tmpim.shape) != 3:
            tmpim = color.gray2rgb(tmpim)
        imlist.append(tmpim)
    return imlist



def injectFloat(floatstring='1.0'):
    return(float(floatstring))

### IMAGE PROCESSING

 
def getRGB(imlist ):
    print('extracting rgb...')
    r_out = []
    g_out = []
    b_out = []
    for image in tqdm.tqdm(imlist):
        [r, g, b] = np.mean(image,axis=(0,1))
        r_out.append(r)
        g_out.append(g)
        b_out.append(b)
    return r_out, g_out, b_out


def getEntropy(imlist ):
    print('extracting entropy...')
    e_out = []
    for image in tqdm.tqdm(imlist):
        imgray = color.rgb2gray(image)
        entr_img = entropy(imgray, disk(10))
        e = np.mean(entr_img.flatten())
        e_out.append(e)
    return e_out
        
#### DATA MANIPULATINO

def reduceDims(vecList):
    my_tsne = manifold.TSNE(n_components=2)
    vecArr = np.asarray(vecList)
    xy = my_tsne.fit_transform(vecList)
    return xy[:,0], xy[:,1]


def toVectors(vecList, floatList1, floatList2):
    width_out = 0
    
    #First, see if 2/3 are none - in that case just return what you get in
    inList = [vecList, floatList1, floatList2]
    areValid = [vv is not None for vv in inList]
    if np.sum(areValid)==1:
        return inList[np.where(areValid)[0][0]]
    
    if vecList is None:
        newVecList = []
        L = len(floatList1)
        for i in range(L):
            newVecList.append(np.asarray([floatList1[i],floatList2[i]]))
    elif floatList1 is None:
        newVecList = []
        L = len(floatList2)
        for i in range(L):
            newVector = np.append(vecList[i], np.asarray(floatList2[i]))
            newVecList.append(newVector)
    elif floatList2 is None:
        newVecList = []
        L = len(floatList1)
        for i in range(L):
            newVector = np.append(vecList[i], np.asarray(floatList1[i]))
            newVecList.append(newVector)
    else:
        newVecList = []
        L = len(floatList1)
        for i in range(L):
            newVector = np.append(vecList[i], np.asarray(floatList1[i]))
            newVector = np.append(newVector, np.asarray(floatList1[i]))
            newVecList.append(newVector)
    return newVecList
        
            

def toList(vecList, indexX, indexY):
    vecList = np.asarray(vecList)
    if indexX is None:
        return vecList[:,indexY].tolist()
    elif indexY is None:
        return vecList[indexX,:].tolist()
    else:
        return vecList[indexX,indexY]

    
    
### OUTPUTS
    
def scatterPlot(x, y):
    print('plotting xy...')
    plt.figure()
    plt.scatter(x, y)
    return 0
            
    
def consolePrint(anyinput):
    print(anyinput)
    return 0


def viewSingleImage(imageList, whichImage):
    if whichImage is None:
        whichImage = 0
    whichImage = int(np.round(float(whichImage)))
    tmpim = imageList[whichImage]
    plt.imshow(tmpim)
    return 0


def pixPlot(imageList, xCoords, yCoords):
    plt.figure(figsize=(15,15))
    imwidth = round((np.max(xCoords) - np.min(xCoords))/ (0.15*len(xCoords)))
    imheight = round((np.max(yCoords) - np.min(yCoords))/ (0.15*len(xCoords)))
    imwidth = np.max([imwidth, 0])
    imheight = np.max([imheight, 0])
    print('plotting images...')
    plt.scatter(xCoords, yCoords)
    for i in tqdm.tqdm(range(len(xCoords))):
        thisim = imageList[i]
        left = xCoords[i]
        right = left+imwidth
        bottom = yCoords[i]
        top = bottom+imheight
        
        plt.imshow(thisim, extent=[left, right, bottom, top])
    plt.xlim([np.min(xCoords), np.max(xCoords)])
    plt.ylim([np.min(yCoords), np.max(yCoords)])
    return 0












