import os
os.chdir("/Users/jonasschmitten/Downloads")

#Import libraries
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.random import seed
from numpy.random import multivariate_normal
import pandas as pd
import statsmodels.api as sm

#Import own functions
import jonas_schmitten_selfstudy_functions as sim

# to ensure replicability
seed(511) 

# Define parameters 
# mean of multivariate normal
mean = [12, 0]             
# diagonal covariance matrix of MVN
cov_mat = [[9, 2], [2, 1]]    
#true coefficients
betas = np.array([5000,100,300,1000])  
#noise variance 
error_var = 80                         
#number of monte carlo simulation
num_simulations = 100              





#first simulation using the first data generating process with treatment 
#variable assigned with binomial probability == 0.6
first_simulation = sim.simulation(betas,mean,cov_mat,error_var,num_simulations,sim.dgp)

#ATE IPW OLS plot
plt.figure(figsize=(15,8))
plt.ylabel('Average Treatment Effect', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","ate_ipw",  data=first_simulation, color = 'blue')
ax2 = sns.lineplot("n","ate_ols",  data=first_simulation, color = 'red')
#draw horizontal line to show where true beta lies (as pre-specified)
ax1.axhline(betas[1], color='black')

#Variance OLS plot
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ols",  data=first_simulation, color = 'red')

#Variance IPW plot
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ipw",  data=first_simulation, color = 'red')





#second simulation using the first data generating process with treatment
#variable assigned with binomial probability == 0.3
second_simulation = sim.simulation(betas,mean,cov_mat,error_var,num_simulations,sim.dgp2)


#ATE OLS plot 
plt.figure(figsize=(15,8))
plt.ylabel('Average Treatment Effect', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","ate_ols",  data=second_simulation, color = 'blue')
#draw horizontal line to show where true beta lies (as pre-specified)
ax1.axhline(100, color='black')

#Variance OLS plot 
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ols",  data=second_simulation, color = 'red')

#ATE IPW plot 
plt.figure(figsize=(15,8))
plt.ylabel('Average Treatment Effect', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","ate_ipw",  data=second_simulation, color = 'blue')
#draw horizontal line to show where true beta lies (as pre-specified)
ax1.axhline(100, color='black')

#Variance IPW plot 
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ipw",  data=second_simulation, color = 'red')





#third simulation using the third data generating process with treatment
#variable assigned with binomial probability == 0.5 and common support violation
#due to scaling of the first coefficient after treatment assignment
third_simulation = sim.simulation(betas,mean,cov_mat,error_var,num_simulations,sim.dgp3)

#ATE OLS plot
plt.figure(figsize=(15,8))
plt.ylabel('Average Treatment Effect', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","ate_ols",  data=third_simulation, color = 'blue')
#draw horizontal line to show where true beta lies (as pre-specified)
ax1.axhline(100, color='black')

#Variance OLS plot 
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ols",  data=third_simulation, color = 'red')

#ATE IPW plot 
plt.figure(figsize=(15,8))
plt.ylabel('Average Treatment Effect', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","ate_ipw",  data=third_simulation, color = 'blue')
#draw horizontal line to show where true beta lies (as pre-specified)
ax1.axhline(100, color='black')

#Variance IPW plot 
plt.figure(figsize=(15,8))
plt.ylabel('Variance', size = 15)
plt.xlabel(' ')
sns.set_style("ticks")
ax1 = sns.lineplot("n","var_ipw",  data=third_simulation, color = 'red')
