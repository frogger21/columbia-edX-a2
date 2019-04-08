## assignment 2 for week 6 columbiaX 
## by jae chang 2019/04/02

from __future__ import division
import numpy as np
import sys
import math

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

##X_train, each row is vector x(i)
##Y_train, each row is in R1 dimension and goes from 0 to 9 (10 classes)
##X_test, same format as x_train

##x_train example
## 1 age(1) height(1) weight(1)
## 1 age(2) height(2) weight(2)
##
##         ...
##
## 1 age(n) height(n) weight(n)
##


##output file desired
## n=1: p(0) p(1)  .... p(9)
##
##   ....
##
## n=n: p(0) p(1) ... p(9)

## can make more functions if required

##begin pluginClassifier
def pluginClassifier(Xtrain = X_train, Ytrain =  y_train, Xtest =  X_test):    
   # this function returns the required output
   #see slide 27 of lecture 7 for more details
   numberofK = 10
   nrowx = Xtrain.shape[0] ##n (number of observations in training set)
   ncolx = Xtrain.shape[1] ##number of columns in X (the dimensions of X)  
   piHat = [0] * numberofK ##store the pi(y) for each k classifier in this case 10
   mewY = np.zeros((numberofK,ncolx)) ##mewY of class conditional density
   sigmaY = np.zeros((numberofK*ncolx,ncolx))
   nyarray = [0] * numberofK
   shifter = ncolx
   numOfTestSet = Xtest.shape[0]
   outputmatrix = np.zeros((numOfTestSet,numberofK)) #n x 10 matrix
   d = ncolx
 
   ##calculate class priors, this is the MLE estimate of pi(y)
   for x in range(numberofK):
      for y in Ytrain:
         if y == x:
            piHat[x] += 1
   piHat =  [x/nrowx for x in piHat]    
  
   ##calculate class conditional density
   ##calculate mew y
   for x in range(numberofK):
      temp = -1
      ny = 0
      for y in Ytrain:
         temp += 1
         if x == y:
            mewY[x,] += Xtrain[temp,]
            ny += 1
            #end of if        
      #end of for y
      nyarray[x] = ny       
      mewY[x,] = [Z / ny for Z in mewY[x,]] #divide each element by Ny #see slide 27 of lecture 7
   #end of for x ...
  
   #calculate sigma 
   for x in range(numberofK):
      temp = -1
      temp2 = np.matrix(mewY[x,]) #1xd
      temp2 = temp2.transpose() #dx1
      newshift = shifter*x
      newershift = shifter*(x+1)
      for y in Ytrain:
         temp += 1
         if x == y:
            temp3 = np.matrix(Xtrain[temp,])
            temp3 = temp3.transpose() #the X
            sigmatemp = np.matmul((temp3-temp2),(temp3-temp2).transpose())
            sigmaY[newshift:newershift,] += sigmatemp
       
      #end of for y
      sigmaY[newshift:newershift,] = [Z / nyarray[x] for Z in sigmaY[newshift:newershift,]]
   #end of for X    

   #plug-in classifier
   for x in range(numOfTestSet):
      experiment1 = 0
      #so we are looping through each element in X test
      for y in range(numberofK):
         #calculate each probability of f(x) for each y in 0 to 9
         tempSigma = sigmaY[shifter*y:(shifter*(y+1)),]
         invSigma = np.linalg.inv(tempSigma)
         detSigma = np.linalg.det(tempSigma)
         testx = np.matrix(Xtest[x,]) #1 x d
         mewx = np.matrix(mewY[y,]) #1 x d
         aaa = np.matmul((testx-mewx),invSigma)
         ccc = (testx-mewx)
         bbb = np.matmul(aaa,ccc.transpose())
         bbb = bbb * (-1/2)
         ddd = np.exp(bbb)
         eee2 = (math.pi*2)**((-d/2))
         eee = eee2*ddd*piHat[y]*(detSigma**(-1/2))
         outputmatrix[x,y] = eee
         experiment1 += eee
      outputmatrix[x,] = [Z/experiment1 for Z in outputmatrix[x,]] #normalizing experiment
   #end for x
   #print("MADE IT HERE2\n")
   print("OUTPUT MATRIX\n")
   print(outputmatrix)
   return(outputmatrix) 
 ##end pluginClassifier

print("BEGIN NEW!\n")
print(X_train)
print("Y")
print(y_train)
final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file