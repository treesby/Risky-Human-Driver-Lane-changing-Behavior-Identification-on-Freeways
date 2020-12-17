# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:38:47 2020

@author: weiminj
"""

import numpy as np
import matplotlib.pyplot as plt


# prompt the user for a training set file name
#data_train_lag_conflict.txt data_test_lag_conflict.txt 
#data_train_lead_conflict.txt data_test_lead_conflict.txt

#data_train_lag_conflict_power.txt data_test_lag_conflict_power.txt
#data_train_lead_conflict_power.txt data_test_lead_conflict_power.txt

training_name = input("Enter the traning set file name: ")    

filein=open(training_name,"r")
# load the training set file
atext=filein.readline()
stext=atext.strip()
acell=stext.split("\t")
m=int(acell[0])
n=int(acell[1])
X=np.zeros([m,n+1])
y=np.zeros([m,1])
for k in range(m):
    atext=filein.readline()
    stext=atext.strip()
    acell=stext.split("\t")
    for j in range(n+1):
        if j==0:
            X[k,j]=1            
        else:
            X[k,j]=float(acell[j-1])
    y[k,0]=float(acell[n])
filein.close()

# scaling features to speed up convergence
X_mean=np.mean(X[:,1:], axis=0)
X_std=np.std(X[:,1:], axis=0)
X[:,1:]=(X[:,1:]-X_mean)/X_std

# print weights and J for the training set
num_interations=4000
w=np.zeros([n+1,1])
reduced_w=np.zeros([n+1,1])
alpha=0.1
lamda=3 # if it is too large, the H will reduce to just w0
O_1m=np.ones([1,m])
ZO_1n=np.ones([1,n+1])
ZO_1n[0][0]=0
regulz=1 # 1 is regularization
threshold=0.4

for iteration in range(num_interations):
       
# with regularization
    if regulz==1:
        H=1/(1+np.exp(-np.dot(X,w)))
        cost=-(y*np.log(H))-(1-y)*np.log(1-H)
        J=np.dot(O_1m,cost)/m+lamda/2/m*np.dot(ZO_1n,w*w)
        if iteration==0:
            print("Initial J= ",J)
        reduced_w[0]=w[0]
        reduced_w[1:]=(1-alpha*lamda/m)*w[1:]
        w=reduced_w-(alpha/m*np.dot((H-y).T,X)).T
    else:
        H=1/(1+np.exp(-np.dot(X,w)))
        cost=-(y*np.log(H))-(1-y)*np.log(1-H)
        J=np.dot(O_1m,cost)/m # without regularization
        w=w-(alpha/m*np.dot((H-y).T,X)).T


    if iteration ==num_interations-1:
        print("Final J= ",J)
        #print(H)
        #print("updated w = ",w)
    if (iteration >=0) and (iteration <=num_interations-1):
        plt.scatter(iteration,J,color="red",marker=".")
i=0
print("final weights are")
for w_i in w:
    print("w"+str(i)+" is",w_i)
    i=i+1

plt.xlabel("iteration")
plt.ylabel("J")
plt.title("J verses iteration")
plt.savefig("Jin_Weimin_MyPlot.png")
plt.show()






def test(w,test_name):
    filein=open(test_name,"r") 
    # load the training set file
    atext=filein.readline()
    stext=atext.strip()
    acell=stext.split("\t")
    m=int(acell[0])
    n=int(acell[1])
    X=np.zeros([m,n+1])
    y=np.zeros([m,1])
    for k in range(m):
        atext=filein.readline()
        stext=atext.strip()
        acell=stext.split("\t")
        for j in range(n+1):
            if j==0:
                X[k,j]=1            
            else:
                X[k,j]=float(acell[j-1])
        y[k,0]=float(acell[n])
    filein.close()
    # must scale features for test dataset also
    X_mean=np.mean(X[:,1:], axis=0)
    X_std=np.std(X[:,1:], axis=0)
    X[:,1:]=(X[:,1:]-X_mean)/X_std
    O_1m=np.ones([1,m])
    H=1/(1+np.exp(-np.dot(X,w)))
    TP=0
    TN=0
    FP=0
    FN=0
    for k in range (m):
        if y[k,0]==0:
            if H[k,0]<threshold:
                TN=TN+1
            else:
                FP=FP+1
        else: 
            if H[k,0]<threshold:
                FN=FN+1
            else:
                TP=TP+1
    cost=-(y*np.log(H))-(1-y)*np.log(1-H)
    if regulz==1:
        J=np.dot(O_1m,cost)/m+lamda/2/m*np.dot(ZO_1n,w*w)
    else:
        J=np.dot(O_1m,cost)/m # without regularization
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1_score=2/(1/precision+1/recall)
    print("J for "+test_name+" is",J)
    print("FP for "+test_name+" is",FP)
    print("TP for "+test_name+" is",TP)
    print("FN for "+test_name+" is",FN)
    print("TN for "+test_name+" is",TN)
    print("accuracy for "+test_name+" is",accuracy)
    print("precision for "+test_name+" is",precision)
    print("recall for "+test_name+" is",recall)
    print("F1_score for "+test_name+" is",F1_score)
             
# prompt the user for a test set file name

test_name = input("Enter the test set file name: ")

test(w,test_name)