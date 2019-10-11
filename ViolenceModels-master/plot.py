import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch 
import pandas as pd
import os

def fromCSV(file):
  df = pd.read_csv(file)
  llist = df['Value'].tolist()
  return llist
  
def loadList(name):
  with open(name, 'rb') as filehandle:
    # read the data as binary data stream
    hist2 = pickle.load(filehandle)
    return hist2
  
def getAverageFromFolds(llist, nepochs):
  arr = np.array(llist)
  arr = np.reshape(arr,(5,nepochs))
  arr = np.mean(arr, axis=0)
  return arr.tolist()
  
  
# print(len(train_lost))
def plotScalarFolds(listF,listL, tepochs,typ, fig2, rows,cols, num):
  # fig2 = plt.figure(figsize=(12,5))
  
  for i in range(0,len(listF),tepochs):
  
    a = fig2.add_subplot(rows, cols, num)
    x_train_acc = np.array(listF[i:i+tepochs])
    
    plt.plot(np.arange(0, tepochs, 1),x_train_acc)
    plt.xlabel('Epoca')
    plt.ylabel('Tasa de Acierto')
    a.set_title(str(typ)+' - Tasa de Acierto')
#     plt.legend(['Iteracion 1', 'Iteracion 2', 'Iteracion 3', 'Iteracion 4', 'Iteracion 5'], loc='upper right',fontsize='medium')
    
    a = fig2.add_subplot(rows, cols, num+1)
    x_train_lost = np.array(listL[i:i+tepochs])
    plt.plot(np.arange(0, tepochs, 1),x_train_lost)
    plt.xlabel('Epoca')
    plt.ylabel('Error')
    a.set_title(str(typ)+' - Error')
    plt.legend(['Iteracion 1', 'Iteracion 2', 'Iteracion 3', 'Iteracion 4', 'Iteracion 5'], loc='upper right',fontsize='medium')
    
#   plt.legend(['Iteracion 1', 'Iteracion 2', 'Iteracion 3', 'Iteracion 4', 'Iteracion 5'], loc='upper right',fontsize='medium')
  # plt.show()
  
def plotScalarCombined(trainlist,testlist, tepochs,title, ylabel, fig2, rows, cols, num):
  # # # # # # fig2 = plt.figure(figsize=(12,5))
  a = fig2.add_subplot(rows, cols, num)

  x = np.arange(0, tepochs, 1)
  plt.plot(x,trainlist,'r')
  plt.plot(x,testlist,'b')
  plt.xlabel('Epoca')
  plt.ylabel(ylabel)
  a.set_title(title)
  plt.legend(['Train', 'Test'], loc='upper right',fontsize='large')
  # plt.show()