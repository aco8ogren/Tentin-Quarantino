
#hyperparameter_optimization
# %%
from skopt import gp_minimize
from skopt.plots import plot_convergence
import numpy as np
import matplotlib.pyplot as plt

# %%
def f(x):
    x1,x2,x3,x4 =  x
    return x1**2-x2*x3+x4*x3-np.sin(x2**x1)-x1

# %%
res = gp_minimize(f,                  # the function to minimize
                                      # the bounds on each dimension of x
                  [
                        (-2.0, 2.0),       # p_err_frac: Parameter error estimate fraction (i.e. .05 --> 5% error)
                        (1,5),             # D_THRES: If a state does not have more than this number of deaths by train_til, we do not make predictions (or, we make cluster predictions)
                        (4,7),             # death_weight: factor by which to weigh error for death data more than symptomatic infected data during SEIIRQD optimization
                        (9, 10)           # alpha: the alpha from LeakyReLU determines how much to penalize the SEIIRQD objective function for over predicting the symptomatic infected
                  ],      
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234,  # the random seed
                  verbose = True)   

# %%
fig = plot_convergence(res)
fig.savefig("bayesian_optimization_convergence.pdf")

