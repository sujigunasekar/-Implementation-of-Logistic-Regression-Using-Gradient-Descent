# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and Load the dataset.

2.Define X and Y array and Define a function for costFunction,cost and gradient.

3.Define a function to plot the decision boundary.

4.Define a function to predict the Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Suji.G
RegisterNumber: 212222230152
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of x:
![Screenshot 2023-09-30 180517](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/5ce4469f-d24e-4b52-95e9-c5eb6e0f4823)
### Array value of y:
![Screenshot 2023-09-30 181600](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/ee286c0c-17b2-48dd-baae-698173d35e4e)
### Score graph:
![Screenshot 2023-09-30 180555](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/2d8bb092-81ae-400d-b41d-f39fccf97fb0)
### Sigmoid function graph:
![Screenshot 2023-09-30 180641](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/365ebaa1-f69d-4a23-8e60-b0abe800c2bd)
### X train grad value:
![Screenshot 2023-09-30 180650](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/b8a1a219-6939-4328-a986-7224735b3fa0)
### Y train grad value:
![Screenshot 2023-09-30 180656](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/df133491-e2b2-4035-bc5b-71fcac17e056)
### Regression value:
![Screenshot 2023-09-30 180710](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/3b2fe8ad-23db-4174-966e-abf5c8721f53)
### decision boundary graph:
![Screenshot 2023-09-30 180721](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/548bd9fc-267f-4be3-8597-160030ca592a)
### Probability value:
![Screenshot 2023-09-30 180728](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/0fc3de56-6c15-49c6-987d-a45d09468b5e)
### Prediction mean value:
![Screenshot 2023-09-30 180733](https://github.com/sujigunasekar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119559822/7df399dd-ca69-445c-be1a-8109b293a265)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

