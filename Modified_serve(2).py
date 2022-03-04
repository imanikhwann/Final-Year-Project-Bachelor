# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 01:06:33 2021

@author: owner
"""



"""
For UTDMHAD dataset (Segmentation difference)
"""
#importing library 

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import array as arr
import statistics as st
from numpy import arange
import xlsxwriter
import os

def all_process(loaded_data):
    #load data file into python
    file_data = scipy.io.loadmat(loaded_data)
    
    #extract data from files into separate array
    data=file_data['d_iner']
    
    k = len(data)
    
    if (k<273): 
        shape = np.shape(data) 
        padded_array = np.zeros((273,6)) 
        padded_array[:shape[0], :shape[1]] = data
    else:
        padded_array = data
        
    x1=padded_array[:,0]
    y1=padded_array[:,1]
    z1=padded_array[:,2]
    x2=padded_array[:,3]
    y2=padded_array[:,4]
    z2=padded_array[:,5]
    
    
    #plot raw signal into graph
    
    plt.plot(x1, 'r--', label='X1')
    plt.plot(y1, 'g--', label='Y1')
    plt.plot(z1, 'b--', label='Z1')
    
    plt.legend()
    plt.show()
    
    plt.plot(x2, 'r--', label='X2')
    plt.plot(y2, 'g--', label='Y2')
    plt.plot(z2, 'b--', label='Z2')
    
    plt.legend()
    plt.show()
    
    #measure the length of the data. All six data should have same length
    d = len(padded_array)
    
    """
    FILTERING
    """
    #provide empty array for each data for low and high pass filtered value
    
    Low_X1 = np.zeros(d)
    High_X1 = np.zeros(d)
    Low_Y1 = np.zeros(d)
    High_Y1 = np.zeros(d)
    Low_Z1 = np.zeros(d)
    High_Z1 = np.zeros(d)
    
    Low_X2 = np.zeros(d)
    High_X2 = np.zeros(d)
    Low_Y2 = np.zeros(d)
    High_Y2 = np.zeros(d)
    Low_Z2 = np.zeros(d)
    High_Z2 = np.zeros(d)
    
    alpha = 0.8 #value for filtering
    
    #replace first value of each filtered array with 0
    Low_X1[0] = 0
    Low_Y1[0] = 0
    Low_Z1[0] = 0
        
    Low_X2[0] = 0
    Low_Y1[0] = 0
    Low_Z1[0] = 0
    
    #low filtering followed by high of X! Y1 Z1
    for i in range (1,d,1): 
    
        Low_X1[i] = alpha * Low_X1[i-1] + (1-alpha) * x1[i]
        Low_Y1[i] = alpha * Low_Y1[i-1] + (1-alpha) * y1[i]
        Low_Z1[i] = alpha * Low_Z1[i-1] + (1-alpha) * z1[i]
            
        High_X1[i] = x1[i] - Low_X1[i]
        High_Y1[i] = y1[i] - Low_Y1[i]
        High_Z1[i] = z1[i] - Low_Z1[i]
    
    #plot data before filter, low filter and high filter to observe the effect
    plt.plot(x1, 'g--', label='X1')
    plt.plot(Low_X1, 'r--', label='X1 LP')
    plt.plot(High_X1, 'b--', label='X1 HP')
        
    plt.legend()
    plt.show()
    
    plt.plot(y1, 'g--', label='Y1')
    plt.plot(Low_Y1, 'r--', label='Y1 LP')
    plt.plot(High_Y1, 'b--', label='Y1 HP')
        
    plt.legend()
    plt.show()
        
    plt.plot(z1, 'g--', label='Z1')
    plt.plot(Low_Z1, 'r--', label='Z1 LP')
    plt.plot(High_Z1, 'b--', label='Z1 HP')
        
    plt.legend()
    plt.show()
    
    #low filtering followed by high of X2 Y2 Z2
    for i in range (1,d,1): 
    
        Low_X2[i] = alpha * Low_X2[i-1] + (1-alpha) * x2[i]
        Low_Y2[i] = alpha * Low_Y2[i-1] + (1-alpha) * y2[i]
        Low_Z2[i] = alpha * Low_Z2[i-1] + (1-alpha) * z2[i]
            
        High_X2[i] = x2[i] - Low_X2[i]
        High_Y2[i] = y2[i] - Low_Y2[i]
        High_Z2[i] = z2[i] - Low_Z2[i]
    
    #plot data before filter, low filter and high filter to observe the effect
    plt.plot(x2, 'g--', label='X2')
    plt.plot(Low_X2, 'r--', label='X2 LP')
    plt.plot(High_X2, 'b--', label='X2 HP')
        
    plt.legend()
    plt.show()
    
    plt.plot(y2, 'g--', label='Y2')
    plt.plot(Low_Y2, 'r--', label='Y2 LP')
    plt.plot(High_Y2, 'b--', label='Y2 HP')
        
    plt.legend()
    plt.show()
        
    plt.plot(z2, 'g--', label='Z2')
    plt.plot(Low_Z2, 'r--', label='Z2 LP')
    plt.plot(High_Z2, 'b--', label='Z2 HP')
        
    plt.legend()
    plt.show()
    
    """
    SEGEMENTATION
    """
    fs= 50
    h = 15
    
    #X1 Y1 Z1 segmentation
    X11 = High_X1[5:268]
    
    Y11 = High_Y1[5:268]

    Z11 = High_Z1[5:268]
    
    #X2 Y2 Z2 Segmentation
    X21 = High_X2[5:268]

    Y21 = High_Y2[5:268]

    Z21 = High_Z2[5:268]

    
    
    """
    FEATURE EXTRACTION
    """
    #feature extraction general function
    def feature_extraction(segmented_data):
        Mean = st.mean(segmented_data)
        Median = st.median(segmented_data)
        print(Mean, Median)
        mini = min(segmented_data)
        maxi = max(segmented_data)
        print(mini, maxi)
        Range1 = maxi - mini
        print(Range1)
        return Mean, Median, mini, maxi, Range1
    
    #calling out feature extraction function for every column of data with return tuples    
    X1W1 = feature_extraction(X11)
    
    Y1W1 = feature_extraction(Y11)
    
    Z1W1 = feature_extraction(Z11) 
    
    X2W1 = feature_extraction(X21) 

    Y2W1 = feature_extraction(Y21) 
    
    Z2W1 = feature_extraction(Z21) 

    
    
    
    #array of data to call in function write
    content = [X1W1, Y1W1, Z1W1, X2W1, Y2W1, Z2W1]
    
    global column, row 
    column = 0
    
    #function to write into excel    
    def write_to_file(thingy):
        global row, column
        for item in thingy:
            worksheet.write(row,column,item)
            column += 1
        column += 0
    
    #for loop to call function for every 'content'
    for things in content:
        write_to_file(things)
    

    

#initializing row of excel
row = 0
column = 0    
directory = r'C:\Users\owner\Desktop\SMBE\FYP\FYP 2\Python\Dataset 2\serve'     

location_file = []

#initializing for data head
head = ['MeanX1W1', 'MedianX1W1', 'MinX1W1', 'MaxX1W1', 'RangeX1W1',
        'MeanY1W1', 'MedianY1W1', 'MinY1W1', 'MaxY1W1', 'RangeY1W1',
        'MeanZ1W1', 'MedianZ1W1', 'MinZ1W1', 'MaxZ1W1', 'RangeZ1W1',
        
        'MeanX2W1', 'MedianX2W1', 'MinX2W1', 'MaxX2W1', 'RangeX2W1',
        'MeanY2W1', 'MedianY2W1', 'MinY2W1', 'MaxY2W1', 'RangeY2W1',
        'MeanZ2W1', 'MedianZ2W1', 'MinZ2W1', 'MaxZ2W1', 'RangeZ2W1']

#calling out excel write module
workbook= xlsxwriter.Workbook('Modified_serve(2).xlsx')
worksheet = workbook.add_worksheet()

#function to write data head
def to_write_head(DATA):
    global row
    column = 0
    for hd in DATA:
        worksheet.write(row,column,hd) 
        column += 1
            
to_write_head(head)
row += 1

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if filename.endswith(".mat"):
        all_process(filename)
        location_file.append(f)
    
    else:
        continue
    row += 1
   
workbook.close() #close workbook after finish    
    
    
    