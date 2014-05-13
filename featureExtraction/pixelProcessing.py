import os
import csv as csv
import numpy as np
from skimage import io
from skimage import color
from skimage.transform import resize
from sklearn.externals import joblib
from PIL import Image

###############################################################
#Variables that might change each run

trainImgs = './TrainingData/images_training_rev1/'
testImgs = './TestData/images_test_rev1/'
inputImgs = trainImgs

outputFilename = './Features/TrainAllFeatures.csv'

numFeatures = 401

#Don't forget to change the file range in the following line - (for f in files[0:1000]:)

###############################################################

###############################################################
#Gets average color feature

def get_average_color((x,y), n, image):
 
    r, g, b = 0, 0, 0
    count = 0
    for s in range(x, x+n+1):
        for t in range(y, y+n+1):
            pixlr, pixlg, pixlb = image[s, t]
            r += pixlr
            g += pixlg
            b += pixlb
            count += 1
    return ((r/count), (g/count), (b/count))

###############################################################

###############################################################
#Gets brightest pixel feature

def get_brightest_pixel((x,y), n, image): 
    brightness = 0
    count = 0
    for s in range(x, x+n+1):
        for t in range(y, y+n+1):
            pixlr, pixlg, pixlb = image[s, t]
            brightPix = pixlr+pixlg+pixlb
            if brightPix > brightness:
              brightness = brightPix
            count += 1
    return brightness


###############################################################

###############################################################

with open(outputFilename, 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  a = [0] * numFeatures
  writer.writerow(a) 
  for path, dirs, files in os.walk(inputImgs):
      #sort file names into numeric order  
      files = sorted(files)
      for f in files[0:10000]:
        galName = np.array(f[:-4])
        path = inputImgs + f
#        img = Image.open(path).load()
#        imgAvgColor = get_average_color((162,162),100,img)
#        averageBrightness = np.array([sum(imgAvgColor)/3])
#        brightestPixel = get_brightest_pixel((162,162),100,img)
        img = io.imread(path)
#        img = io.imread(path, as_gray=True)
        cropped = img[137:287,137:287]
        gimg = color.colorconv.rgb2grey(cropped)
        resized = resize(gimg, (20,20))
        i = np.vstack(resized)
        flat = i.flatten()
        total = np.append(galName, flat)
#        total = np.append(galName, np.append(imgAvgColor, np.append(averageBrightness, brightestPixel)))
        writer.writerow(total)
