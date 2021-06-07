import numpy as np
import seaborn as sns
from numpy.random import multivariate_normal
import pandas as pd
import statsmodels.api as sm

#############################################################################################
# Data Generating Process Functions 
#############################################################################################

def dgp(b,mv_mean,mv_cov,u_sd,n):
    """
    data generating process multivariate exogenous variables and binomial
    treatment variable with treatment assignment = 0.5
    
    b = pre-specified correct betas 
    mv_mean = pre-specified correct mean of exogenous variables
    mv_cov = pre-specified covariance variance matrix
    us_sd = pre-specified error variance (deterministic)
    n = sample size on which data is generated
    """   
    #OLS better
    x = multivariate_normal(mv_mean, mv_cov, n)
    #add treatment variable
    x = np.c_[np.random.binomial(1, 0.5, size=n),x]
    #outcome variable
    y = b[0] + x @ b[1:] + np.random.normal(0,u_sd,n)  
    
    return(x,y)

def dgp2(b,mv_mean,mv_cov,u_sd,n):
    """
    data generating process multivariate exogenous variables and binomial
    treatment variable with treatment assignment = 0.1
    
    b = pre-specified correct betas 
    mv_mean = pre-specified correct mean of exogenous variables
    mv_cov = pre-specified covariance variance matrix
    us_sd = pre-specified error variance (deterministic)
    n = sample size on which data is generated
    """   
    #IPW better
    x = multivariate_normal(mv_mean, mv_cov, n)
    #add treatment variable
    x = np.c_[np.random.binomial(1, 0.1, size=n),x]
    #outcome variable
    y = b[0] + x @ b[1:] + np.random.normal(0,u_sd,n)  
   
    return(x,y)

def dgp3(b,mv_mean,mv_cov,u_sd,n):
    """
    data generating process that violates the common support assumption

    b = pre-specified correct betas 
    mv_mean = pre-specified correct mean of exogenous variables
    mv_cov = pre-specified covariance variance matrix
    us_sd = pre-specified error variance (deterministic)
    n = sample size on which data is generated
    """    
    #Violation 
    x = multivariate_normal(mv_mean, mv_cov, n)
    #add treatment variable
    x = np.c_[np.random.binomial(1, 0.5, size=n),x]
    #scale up first exogenous variable to break common support 
    #assumption - full separation not possible, tweak scalar
    x[:,1] = np.where(x[:,0] == 1,x[:,1]*1.5,x[:,1])
    y = b[0] + x @ b[1:] + np.random.normal(0,u_sd,n)  
    
    return(x,y)

#############################################################################################
# Estimator functions OLS and IPW
#############################################################################################

def ols(x,y):
    """
    function to calculate the ordinary least squares estimator 

    x = exogenous and treatment variable matrix from the data generating process
        without constant
    y = outcome variable from the data generating process
    """
    # add constant
    x_c = np.c_[np.ones(y.shape[0]),x] 
    # calculate betas
    betas = np.linalg.inv(x_c.T @ x_c) @ x_c.T @ y 

    return(betas)

def IPW(x,y):
    """
    function to compute the inverse probability weighting estimator

    x = exogenous and treatment variable matrix from the data generating process
    y = outcome variable from the data generating process
    """
    #predict pscores using logistic regression, transform inputs to work with logit function
    pscores = sm.Logit(endog=pd.Series(x[:,0]),exog=sm.add_constant(pd.DataFrame(x[:,1:]))).fit(disp=0).predict()
    #calculate average treatement effect
    ate = np.mean((pd.Series(x[:,0]) * pd.Series(y)) / pscores - ((1 - pd.Series(x[:,0])) * pd.Series(y)) / (1 - pscores))
    
    return(ate)

#############################################################################################
# Simulation function 
#############################################################################################

def simulation(betas, mean, cov_mat, error_var, sims, dgp):
    """
    main function to simulate the estimator using different sample sizes

    betas = pre-specified correct betas 
    mean = pre-specified correct mean of exogenous variables
    cov_mat = pre-specified covariance variance matrix
    error_var = pre-specified error variance (deterministic)
    sims = number of monte carlo simulations in each sample sizes
    dgp = type of data generating process used as inputs for ols and ipw
    """
    #create a results dataframe to store wanted variable
    results = pd.DataFrame(columns=['var_ols','bias_ols','ate_ols', 'var_ipw', 'bias_ipw', 'ate_ipw'])
    for i in range(100,5001,100):
        #create empty lists for ols and ipw coefficients
        ols_list, ipw_list = [], []
        #monte carlo simulation 
        for j in range(sims):
            #generate the data based on the data generating process specified in arguments
            x,y = dgp(betas, mean, cov_mat, error_var, i) 
            #append empty lists with coefficients
            ols_list.append(ols(x,y))
            ipw_list.append(IPW(x,y))

        #calculate ate, variance, and bias of ipw 
        ate_ipw = np.mean(np.array(ipw_list))
        var_ipw = np.mean((np.array(ipw_list) - np.mean(np.array(ipw_list)))**2)
        bias_ipw = np.mean(np.array(ipw_list)- betas[1])

        #calculate ate, variance, and bias of ols 
        ate_ols = np.mean(np.vstack(ols_list)[:,1])
        var_ols = np.mean((np.array(ols_list)[:,1] - np.mean(np.array(ols_list)[:,1]))**2)
        bias_ols = np.mean(np.array(ols_list)[:,1]- betas[1])
        
        #store all results in the results dataframe created at the top of the function
        results.loc[int(i/100)-1] =  [var_ols,bias_ols,ate_ols, var_ipw, bias_ipw, ate_ipw]
    
    #add column for sample size 
    results.insert(0, "n", range(100,5001,100))
    
    return(results)