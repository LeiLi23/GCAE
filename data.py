#!/usr/bin/python
# -*- coding: UTF-8 -*-
# encoding: GBK

import xlrd #读excel文件
from math import e #导出e
import numpy as np #进行矩阵的操作
import math

M = 495; N = 383;K_CONECT = 5430
Y = np.zeros((M ,N))
S_r = np.zeros((M ,M))
S_d = np.zeros((N ,N))      #声明三个矩阵

f_DSEMANTIC_1 = open(r'.\Data\disease semantic similarity1.txt')
f_DSEMANTIC_2 = open(r'.\Data\disease semantic similarity2.txt')
f_RFUNCTIONAL = open(r'.\Data\miRNA functional similarity.txt')
f_DN = xlrd.open_workbook(r'.\Data\diseasenumber.xlsx')
f_RN = xlrd.open_workbook(r'.\Data\miRNAnumber.xlsx')
fw_PredictResult= open(".\predictedresult.txt","w")     #调用数据


F1=[]
lines_1=f_DSEMANTIC_1.readlines() #读出文件所有行
for i in range(N):
    line_1=[float(x) for x in lines_1[i].split()]   #以空格为符号进行分解
    F1.append(line_1)    
dSemanticArray_1 = np.array(F1)  #以数组的形式存储

F2=[]
lines_2=f_DSEMANTIC_2.readlines()
for j in range(N):
    line_2=[float(x) for x in lines_2[j].split()]
    F2.append(line_2)
dSemanticArray_2 = np.array(F2)

F3=[]
lines_3=f_RFUNCTIONAL.readlines()
for j in range(M):
    line_3=[float(x) for x in lines_3[j].split()]
    F3.append(line_3)
rFunctionalArray = np.array(F3)

dSemanticArray = (dSemanticArray_1+dSemanticArray_2)/2


f_DWeight = open(r'.\Data\disease semantic similarity weight.txt')  #1有关联 0无关联
FDW=[]
linesDW=f_DWeight.readlines() 
for i in range(N):
    lineDW=[float(x) for x in linesDW[i].split()]  #以空格为界限分开
    FDW.append(lineDW)
dWeightArray = np.array(FDW)

f_RWeight = open(r'.\Data\miRNA functional similarity weight.txt')
FRW=[]
linesRW=f_RWeight.readlines()
for i in range(M):
    lineRW=[float(x) for x in linesRW[i].split()]
    FRW.append(lineRW)
rWeightArray = np.array(FRW)

f_MDI = xlrd.open_workbook('.\Data\miRNA-disease.xlsx')
KnownAssociation = []
for i in range(K_CONECT):   #K_CONECT = 5430
    rnarow = int(f_MDI.sheet_by_name('miRNA-disease').cell_value(i,0))-1  #通过名称获取，返回单元格的数据
    disrow = int(f_MDI.sheet_by_name('miRNA-disease').cell_value(i,1))-1
    Y[rnarow][disrow] = 1
    KnownAssociation.append([disrow,rnarow])  #将5430个联系存储起来

Gama_d1 = 1.

GuassD=np.zeros((N,N))
Gama_d=N*Gama_d1/sum(sum(Y * Y)) 
for i in range(N):
    for j in range(N):
        GuassD[i][j]=e**(-Gama_d*(np.dot(Y[:,i]-Y[:,j],Y[:,i]-Y[:,j])))

Gama_m1 = 1.

GuassR=np.zeros((M,M))
Gama_m=M*Gama_m1/sum(sum(Y * Y)) 
for i in range(M):
    for j in range(M):
        GuassR[i][j]=e**(-Gama_m*(np.dot(Y[i,:]-Y[j,:],Y[i,:]-Y[j,:])))  #m和d的高斯内核相似性
for i in range(N):
    for j in range(N):
        if dWeightArray[i][j] == 1:
            S_d[i][j] = (dSemanticArray[i][j]+GuassD[i][j])/2
        elif dWeightArray[i][j] == 0:
            S_d[i][j] = GuassD[i][j]

for i in range(M):
    for j in range(M):
        if rWeightArray[i][j] == 1:
            S_r[i][j] = (rFunctionalArray[i][j]+GuassR[i][j])/2
        elif rWeightArray[i][j] == 0:
            S_r[i][j] = GuassR[i][j]

