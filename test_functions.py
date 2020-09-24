## Preparation
import numpy as np
import numpy.matlib
from sklearn.linear_model import LassoCV
import gurobipy as gp
from gurobipy import GRB

########################################
## Function: Generating Gaussian Samples
def gnrt_normal(n,p,rho,snr,beta0):
    sigma=np.matlib.zeros((p,p))
    for i in range(p):
        for j in range(p):
            sigma[i,j]=rho**(abs(i-j))
    X=np.random.multivariate_normal(mean=np.zeros(p),cov=sigma,size=n)
    var=float(beta0.T.dot(sigma).dot(beta0)/snr)
    I=np.identity(n)
    Y=np.random.multivariate_normal(mean=np.array(X.dot(beta0).T)[0],cov=var*I,size=1)
    return np.matrix(X),np.matrix(Y.T)

##############################################################
## Function: Solving Best Subset Selection with Gurobi package
def solve_dis_alg(Y,X,epsilon,k):
    p=X.shape[1]
    L=np.linalg.eig(X.T.dot(X))[0][0]
    beta_old=np.matrix(np.zeros(p)).T
    beta_new=np.matrix(np.array(list(np.ones(k))+list(np.zeros(p-k)))).T
    while np.linalg.norm(beta_old-beta_new)>epsilon:
        beta_old=beta_new
        beta_new=np.zeros(p)
        T=np.array(beta_old-1/L*(X.T.dot(X).dot(beta_old)-X.T.dot(Y))).T[0]
        tt_sort=-np.sort(-abs(T))
        for i in range(k):
            for j in range(p):
                if tt_sort[i]==T[j]:
                    beta_new[j]=T[j]
                    break
        beta_new=np.matrix(beta_new).T
    A=[np.flatnonzero(beta_new)]
    active=np.array(np.linalg.inv(X[:,A[0]].T.dot(X[:,A[0]])).dot(X[:,A[0]].T).dot(Y).T)[0]
    beta_warmstart=np.zeros(p)
    j=0
    for i in A:
        beta_warmstart[i]=active[j]
        j=j+1
    return beta_warmstart

#def solve_bestsubset(Y,X,epsilon,k,type= True):
#    beta_warmstart=solve_dis_alg(Y,X,epsilon,k)
#    if type:
#    else:

        
##########################
## Function: Solving Lasso

def solve_lasso(Y,X):
    model=LassoCV()
    model.fit(X,Y)
    return model.coef_

##############################
## Function: Metrics: FDR, TPR

def FDR(betahat,beta0):
    p=len(list(beta0))
    score=0
    for i in range(p):
        if betahat[i]!=0 and beta0[i]==0:
            score=score+1
    return score/(np.flatnonzero(betahat).shape[0])

def TPR(betahat,beta0):
    p=len(list(beta0))
    score=0
    for i in range(p):
        if betahat[i]!=0 and beta0[i]!=0:
            score=score+1
    return score/(np.flatnonzero(beta0).shape[0])

beta0=np.matrix(list(np.ones(2))+list(np.zeros(2))).T
X,Y=gnrt_normal(7,4,0.35,10**0.05,beta0)
print(solve_dis_alg(Y,X,0.001,2))
print(solve_lasso(Y,X))
