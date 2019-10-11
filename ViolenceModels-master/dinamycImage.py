import numpy as np
from PIL import Image


# class DinamycImage():
#     def __init__(self,):

def getDynamicImage(frames):
    seqLen = len(frames)
    frames = np.stack(frames, axis=0)

    fw = np.zeros(seqLen)  
    for i in range(seqLen): #frame by frame
      fw[i] = np.sum( np.divide((2*np.arange(i+1,seqLen+1)-seqLen-1) , np.arange(i+1,seqLen+1))  )

    fwr = fw.reshape(seqLen,1,1,1)
    sm = frames*fwr
    sm = sm.sum(0)
    sm = sm - np.min(sm) 
    sm = 255 * sm /np.max(sm) 
    img = sm.astype(np.uint8)
    ##to PIL image
    imgPIL = Image.fromarray(np.uint8(img))
    return imgPIL, img

