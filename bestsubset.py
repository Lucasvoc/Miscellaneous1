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
    var=beta0.T.dot(sigma).dot(beta0)/snr
    I=np.identity(n)
    Y=np.random.multivariate_normal(mean=X.dot(beta0),cov=var*I,size=1)
    return X,Y

##############################################################
## Function: Solving Best Subset Selection with Gurobi package
def solve_dis_alg(Y,X,epsilon,k):
    p=X.shape[1]
    L=np.linalg.eig(X.T.dot(X))[0][0]
    beta_old=np.zeros(p)
    beta_new=np.array(list(np.ones(k))+list(np.zeros(p-k)))
    while np.linalg.norm(beta_old-beta_new)>epsilon:
        beta_old=beta_new
        beta_new=np.zeros(p)
        T=beta_old-1/L*(X.T.dot(X).dot(beta_old)-X.T.dot(Y))
        tt_sort=-np.sort(-abs(T))
        j=1
        for i in range(p):
            if T[i]==tt_sort[j]:
                beta_new[i]=T[i]
                if j>k:
                    break
    A=[np.flatnonzero(beta_new)]
    return np.linalg.inv(X[:,A].T.dot(X[:,A])).dot(X.T).dot(Y)

def solve_bestsubset(Y,X,epsilon,k,type= True):
    beta_warmstart=solve_dis_alg(Y,X,epsilon,k)
    if type:
    else:

        
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

######################
## Set SNR(log) series
snr=[0.05,0.09,0.14,0.25,0.42,0.72,1.22,2.07,3.52,6.00]

#####################
#####################
#### Low setting ####
#####################
#####################

####################
## Lasso v.s. BSS ##
####################

# Set beta0
beta01=np.array(list(np.ones(5))+list(np.zeros(5)))
# Vectors to store performances:
score11_lasso_FDR=np.zeros(10)
score11_lasso_TPR=np.zeros(10)
score11_bestsubset_FDR=np.zeros(10)
score11_bestsubset_TPR=np.zeros(10)
# Within iteration, Genrate samples and store their performances
for i in range(10):
    X,Y=gnrt_normal(100,10,0.35,10**snr[i],beta01)
    beta_lasso=solve_lasso(Y,X)
    beta_bestsubset=solve_bestsubset(Y,X,0.001,5,type=True)
    score11_lasso_FDR[i]=FDR(beta_lasso,beta01)
    score11_lasso_TPR[i]=TPR(beta_lasso,beta01)
    score11_bestsubset_FDR[i]=FDR(beta_bestsubset,beta01)
    score11_bestsubset_TPR[i]=TPR(beta_bestsubset,beta01)
print(score11_lasso_FDR,'\n',score11_lasso_TPR,'\n',score11_bestsubset_FDR,'\n',score11_bestsubset_TPR)

#######################################################
## Low setting: BSS with different k (SNR(log)=0.05) ##
#######################################################

# Generate test data 
X,Y=gnrt_normal(100,10,0.35,snr[0],beta01)
# Different k
k=[1,2,3,4,5,7,9]
# Vectors to store performances:
score12_FDR=np.zeros(7)
score12_TPR=np.zeros(7)
# With iteration, test performances with changing k:
for i in range(7):
    beta_bestsubset=solve_bestsubset(Y,X,0.001,k[i],type=True)
    score12_FDR[i]=FDR(beta_bestsubset,beta01)
    score12_TPR[i]=TPR(beta_bestsubset,beta01)
print(score12_FDR,'\n',score12_TPR)

#######################################################
## Low setting: BSS with different k (SNR(log)=6.00) ##
#######################################################

# Generate test data 
X,Y=gnrt_normal(100,10,0.35,snr[9],beta01)
# Different k
k=[1,2,3,4,5,7,9]
# Vectors to store performances:
score13_FDR=np.zeros(7)
score13_TPR=np.zeros(7)
# With iteration, test performances with changing k:
for i in range(7):
    beta_bestsubset=solve_bestsubset(Y,X,0.001,k[i],type=True)
    score13_FDR[i]=FDR(beta_bestsubset,beta01)
    score13_TPR[i]=TPR(beta_bestsubset,beta01)
print(score13_FDR,'\n',score13_TPR)

########################
########################
#### High-5 setting ####
########################
########################

####################################
## High-5 setting: Lasso v.s. BSS ##
####################################
# Set beta0
beta03=np.array(list(np.ones(5))+list(np.zeros(995)))
# Vectors to store performances:
score31_lasso_FDR=np.zeros(10)
score31_lasso_TPR=np.zeros(10)
score31_bestsubset_FDR=np.zeros(10)
score31_bestsubset_TPR=np.zeros(10)
# Within iteration, Genrate samples and store their performances
for i in range(10):
    X,Y=gnrt_normal(50,1000,0.35,10**snr[i],beta03)
    beta_lasso=solve_lasso(Y,X)
    beta_bestsubset=solve_bestsubset(Y,X,0.001,5,type=True)
    score31_lasso_FDR[i]=FDR(beta_lasso,beta01)
    score31_lasso_TPR[i]=TPR(beta_lasso,beta01)
    score31_bestsubset_FDR[i]=FDR(beta_bestsubset,beta01)
    score31_bestsubset_TPR[i]=TPR(beta_bestsubset,beta01)
print(score31_lasso_FDR,'\n',score31_lasso_TPR,'\n',score31_bestsubset_FDR,'\n',score31_bestsubset_TPR)

####################################################
## High-5 setting: BSS with diff k(SNR(log)=0.05) ##
####################################################
# Generate test data 
X,Y=gnrt_normal(50,1000,0.35,snr[0],beta03)
# Different k
k=[1,2,3,4,5,7,9]
# Vectors to store performances:
score32_FDR=np.zeros(7)
score32_TPR=np.zeros(7)
# With iteration, test performances with changing k:
for i in range(7):
    beta_bestsubset=solve_bestsubset(Y,X,0.001,k[i],type=True)
    score32_FDR[i]=FDR(beta_bestsubset,beta01)
    score32_TPR[i]=TPR(beta_bestsubset,beta01)
print(score32_FDR,'\n',score32_TPR)

####################################################
## High-5 setting: BSS with diff k(SNR(log)=6.00) ##
####################################################
# Generate test data 
X,Y=gnrt_normal(50,1000,0.35,snr[9],beta01)
# Different k
k=[1,2,3,4,5,7,9]
# Vectors to store performances:
score33_FDR=np.zeros(7)
score33_TPR=np.zeros(7)
# With iteration, test performances with changing k:
for i in range(7):
    beta_bestsubset=solve_bestsubset(Y,X,0.001,k[i],type=True)
    score33_FDR[i]=FDR(beta_bestsubset,beta01)
    score33_TPR[i]=TPR(beta_bestsubset,beta01)
print(score33_FDR,'\n',score33_TPR)