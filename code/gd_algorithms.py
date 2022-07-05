import numpy as np
import pandas as pd
from keras.datasets import mnist
from numpy.linalg import norm
from numpy import linalg as LA
from matplotlib import pyplot as plt
import warnings
import random
  
# suppress warnings
warnings.filterwarnings('ignore')


##################### LOGISTIC REGRESSION FUNCTIONS ######################

def regularized_logistic_objective(trainX, trainY, u_hat, lmbd):
    
    '''
    Objective function for the regularized logistic method
    
    inputs:
        - trainX:
        - trainY: 
        - u_hat:
        - lmbd:

    outputs:
        - fu:
    '''
    
    classes = np.unique(trainY)
    u = u_hat[:-1]
    u_0 = u_hat[-1]
    
    if len(classes) != 2:
        print(f"The response variable is not binary, it has {len(classes)} different classifications")
    else:
        
        # calculate the three terms of the objective function
        fu_1 = -np.sum(u @ trainX[trainY==classes[1]].T + u_0)   # only for class 1
        fu_2 = np.sum(np.log(1+np.exp(u @ trainX.T + u_0)))
        fu_3 = lmbd*(u_hat.T @ u_hat)
        fu = fu_1 + fu_2 + fu_3
        
        return fu
    

def regularized_logistic_gradient(trainX, trainY, u_hat, lmbd):
    
    '''
    Gradient for the regularized logistic function
    
    inputs:
        - trainX:
        - trainY: 
        - u_hat:
        - lmbd:

    outputs:
        - gradient:
    '''
    
    classes = np.unique(trainY)
    u = u_hat[:-1]
    u_0 = u_hat[-1]
    
    if len(classes) != 2:
        print(f"The response variable is not binary, it has {len(classes)} different classifications")
    else:
        
        diffu_1 = -np.sum(trainX[trainY==classes[1]], axis=0)
        aux = np.exp(u @ trainX.T + u_0)
        diffu_2 = np.sum((aux/(1+aux))[:, None] * trainX, axis=0)
        diffu_3 = 2*lmbd*u
        
        diffu = diffu_1 + diffu_2 + diffu_3
        
        
        diffu0_1 = -np.sum(trainY==classes[1])
        diffu0_2 = np.sum((aux/(1+aux)))
        diffu0_3 = 2*lmbd*u_0
        
        diffu_0 = diffu0_1 + diffu0_2 + diffu0_3
        
        gradient = np.append(diffu, diffu_0)
        
        return gradient
    

def regularized_logistic_gradient_batch(trainX, trainY, u_hat, lmbd, 
                                        batch_indexes):
    
    '''
    Gradient for the regularized logistic function
    
    inputs:
        - trainX:
        - trainY: 
        - u_hat:
        - lmbd:
        - batch_indexes

    outputs:
        - gradient:
    '''
    
    classes = np.unique(trainY)
    u = u_hat[:-1]
    u_0 = u_hat[-1]
    
    if len(classes) != 2:
        print(f"The response variable is not binary, it has {len(classes)} different classifications")
    else:
        if batch_indexes == "full":
            batch_indexes = np.full(trainX.shape[0], True, dtype=bool)
        trainX_batch = trainX[batch_indexes, :]
        trainY_batch = trainY[batch_indexes]
        
        diffu_1 = -np.sum(trainX_batch[trainY_batch==classes[1]], axis=0) 
        aux = np.exp(u @ trainX_batch.T + u_0)
        diffu_2 = np.sum((aux/(1+aux))[:, None] * trainX_batch, axis=0)
        diffu_3 = 2*lmbd*u
        
        diffu = diffu_1 + diffu_2 + diffu_3
        
        
        diffu0_1 = -np.sum(trainY_batch==classes[1])
        diffu0_2 = np.sum((aux/(1+aux)))
        diffu0_3 = 2*lmbd*u_0
        
        diffu_0 = diffu0_1 + diffu0_2 + diffu0_3
        
        gradient = np.append(diffu, diffu_0)
        
        return gradient
    
    
##################### GRADIENT DESCENT ALGORITHMS ######################
    
def gradient_descent_algorithm(gradf, z_0, L, tol, maxit):
    
    '''
    Gradient descent algorithm 
    '''
    
    z_k = z_0
    gradient = gradf(x=z_k, batch_indexes="full")
    i = 0
    grad_hist = []
    for numit in range(maxit):
        norm_grad = norm(gradient)
        grad_hist.append(norm_grad)
        if norm_grad > tol:
            z_k1 = z_k - 1/L * gradient
            z_k = z_k1
            gradient = gradf(x=z_k, batch_indexes="full")
            i += 1
        else:
            return z_k, i, grad_hist
        
    print(f'The algorithm reached the maximum number of iterations: {maxit}')
    return z_k, maxit+1, grad_hist    

    
def accelerated_gradient_algorithm(gradf, L, l, x_0, tol, maxit, alpha_0=0.5):
    
    '''
    Accelerated gradient algorithm
    
    inputs:
        - gradf: function to compute the gradient
        - L: 
        - l:
        - x_0:
        - tol:
        - maxit:
        - alpha_0:
    outputs:
        - yk:
        - numit:
        - grad_hist: list of norm of the gradient
                     for each iteration.
    '''
    
    x_k = x_0
    y_k = x_0
    gradient = gradf(x=y_k, batch_indexes="full")
    alpha_k = alpha_0
    grad_hist = []

    for numit in range(maxit):
        norm_grad = norm(gradient)
        grad_hist.append(norm_grad)
        if norm_grad > tol:
            x_kplus1 = y_k - gradient/L
            coeff = [1, (alpha_k**2 - l/L), -alpha_k**2]
            alpha_kplus1 = max(np.roots(coeff))
            beta_k = (alpha_k * (1 - alpha_k)) / (alpha_k**2 + alpha_kplus1)
            y_kplus1 = x_kplus1 + beta_k * (x_kplus1 - x_k)    
            
            x_k = x_kplus1
            alpha_k = alpha_kplus1
            y_k = y_kplus1
            gradient = gradf(x=y_k, batch_indexes="full")
        else:
            return y_k, numit, grad_hist
    
    print(f'The algorithm reached the maximum number of iterations: {maxit}')
    return y_k, maxit+1, grad_hist


def conjugate_gradient_algorithm(gradf, objf, x_0, tol, maxit, 
                                 line_search):
    
    '''
    Accelerated gradient algorithm
    
    inputs:
        - gradf: function to compute the gradient
        - x_0:
        - tol:
        - maxit:
        - alpha_0:
    outputs:
        - yk:
        - numit:
        - grad_hist: list of norm of the gradient
                     for each iteration.
    '''
    
    x_k = x_0
    g_k = gradf(x=x_k, batch_indexes="full")
    p_k = -g_k
    grad_hist = []
    
    for numit in range(maxit):
        norm_grad = norm(g_k)
        grad_hist.append(norm_grad)
        if norm_grad > tol:
            alpha_k = line_search(objf, x_k, p_k)
            x_kplus1 = x_k + alpha_k*p_k
            g_kplus1 = gradf(x=x_kplus1, batch_indexes="full")
            beta_kplus1 = (g_kplus1 @ g_kplus1)/(g_k @ g_k)
            p_kplus1 = -g_kplus1 + beta_kplus1*p_k  
            
            x_k = x_kplus1
            g_k = g_kplus1
            p_k = p_kplus1
        else:
            return x_k, numit, grad_hist
    
    print(f'The algorithm reached the maximum number of iterations: {maxit}')
    return x_k, maxit+1, grad_hist


def stochastic_gradient_algorithm(gradf, x_0, n,
                                  batch_size, stepsize,
                                  tol, maxit, 
                                  objf=None, line_search=None,
                                  eta=None):
    
    '''
    Stochastic Gradient Descent algorithm 
    
    inputs:
        - gradf: gradient of the objective function
        - x_0: initial value of optimizer 
        - n: number of features in the dataset
        - batch_size:
        - stepsize: method to choose the stepsize, supports 
                    either "backtrack_linesearch" or "constant"
        - tol: convergence tolerance
        - maxit: maximum iterations
        - objf: objective function, must be non-None if the 
                stepsize method is any linesearch.
        - eta: stepsize constant value, must be non-None if 
                the stepsize method is "constant"

    outputs:
        - x_k: optimizer
        - numit: number of iterations
        - grad_hist: list of norm of the gradient
                     for each iteration.
    '''
    
    x_k = x_0
    num_of_batches = int(np.floor(n / batch_size))
    grad_hist = []
    
    perm_indexes = [i for i in range(n)]
    random.shuffle(perm_indexes)
    
    if stepsize == "line_search":   # objf must be non-None
        gradient = gradf(x_k, "full")
        for numit in range(maxit): 
            norm_grad = norm(gradient)
            grad_hist.append(norm_grad)
            p_k = -gradient
            eta_k = line_search(objf, x_k, p_k)
            if norm_grad > tol:
                for i in range(num_of_batches):
                    start_index = (i-1)*batch_size
                    end_index = start_index + batch_size
                    batch_indexes = perm_indexes[start_index:end_index]
                    gradient = gradf(x_k, batch_indexes)
                    x_k1 = x_k - eta_k * gradient
                    x_k = x_k1
            else:
                return x_k, numit, grad_hist

        print(f'The algorithm reached the maximum number of iterations: {maxit}')
        return x_k, maxit+1, grad_hist
    
    elif stepsize == "constant":   # eta must be non-None
        gradient = gradf(x_k, "full")
        for numit in range(maxit):
            norm_grad = norm(gradient)
            grad_hist.append(norm_grad)
            if norm_grad > tol:
                for i in range(num_of_batches):
                    start_index = (i-1)*batch_size
                    end_index = start_index + batch_size
                    batch_indexes = perm_indexes[start_index:end_index]
                    gradient = gradf(x_k, batch_indexes)
                    x_k1 = x_k - eta * gradient
                    x_k = x_k1
            else:
                return x_k, numit, grad_hist

        print(f'The algorithm reached the maximum number of iterations: {maxit}')
        return x_k, maxit+1, grad_hist 
    

##################### OPTIMIZATION SOLVER ###################### 
    
def logistic_regression_optimization(trainX, trainY, lmbd, tol, maxit, gd_algorithm, 
                                     batch_size="full", stepsize="constant", 
                                     line_search=None,
                                     u_hat_init=None):
    
    '''
    Optimize the logistic regression problem for the given data and 
    gradient descent algorithm
    
    inputs:
        - trainX:
        - trainY: 
        - lmbd:
        - tol:
        - maxit:
        - gd_algorithm:
        - batch_size:
        - stepsize:
        - u_hat_init:

    outputs:
        - u_hat:
        - iterations:
    '''
    
    if u_hat_init is None:
        u = np.zeros(trainX.shape[1])
        u_0 = 0
        u_hat_init = np.append(u, u_0)
        
    rl_objective = lambda x: regularized_logistic_objective(trainX, 
                                                            trainY, 
                                                            x, 
                                                            lmbd)
    
    rl_gradient = lambda x, batch_indexes: regularized_logistic_gradient_batch(trainX, 
                                                                               trainY, 
                                                                               x, 
                                                                               lmbd, 
                                                                               batch_indexes)
    
    if gd_algorithm == gradient_descent_algorithm:
        
        L = np.sum(np.einsum('ij, ij->i', 
                             trainX, trainX) + 1)/4 + lmbd   # np.einsum does row by row dot product
        u_hat, iterations, grad_hist = gd_algorithm(rl_gradient, 
                                                    u_hat_init, 
                                                    L, tol, maxit)
    
    elif gd_algorithm == accelerated_gradient_algorithm:
        
        l = lmbd
        L = np.sum(np.einsum('ij, ij->i', 
                             trainX, trainX) + 1)/4 + lmbd   # np.einsum does row by row dot product
        u_hat, iterations, grad_hist = gd_algorithm(rl_gradient, 
                                                    L, l, u_hat_init, 
                                                    tol, maxit, alpha_0=0.5)
            
    elif gd_algorithm == conjugate_gradient_algorithm:
        u_hat, iterations, grad_hist = gd_algorithm(rl_gradient, rl_objective, 
                                                    u_hat_init, tol, maxit, line_search)
    
    elif gd_algorithm == stochastic_gradient_algorithm:
        if stepsize == "constant":
            L = np.sum(np.einsum('ij, ij->i', 
                                 trainX, trainX) + 1)/4 + lmbd   # np.einsum does row by row dot product
            u_hat, iterations, grad_hist = gd_algorithm(rl_gradient, u_hat_init, 
                                                        len(trainY), batch_size, 
                                                        stepsize, tol, maxit, eta=1/L)
        elif stepsize == "line_search":
            u_hat, iterations, grad_hist = gd_algorithm(rl_gradient, u_hat_init, 
                                                        len(trainY), batch_size, 
                                                        stepsize, tol, maxit, 
                                                        objf=rl_objective,
                                                        line_search=line_search)

            
    else:
        print(f"{gd_algorithm} it's not a supported gradient descent algorithm")
    
    return u_hat[:-1], u_hat[-1], iterations, grad_hist


##################### HELPER FUNCTIONS ######################

def backtrack_line_search(fun, x_k, p_k, alpha_init=1, tau=0.5, beta=None):
    
    """
    Backtrack Line Search method for the stepsize alpha^k
    """
    
    alpha_l = alpha_init
    while fun(x_k+alpha_l*p_k) > fun(x_k):
        alpha_l = alpha_l*tau
    alpha_k = alpha_l
    return alpha_k


def armijo_line_search(fun, x_k, p_k, alpha_init=1, tau=0.5, beta=0.01):
    
    """
    Backtrack Line Search method for the stepsize alpha^k
    """
    g_k = -p_k
    alpha_l = alpha_init
    while fun(x_k+alpha_l*p_k) > fun(x_k) + alpha_l * beta * g_k.T @ p_k:
        alpha_l = alpha_l*tau
    alpha_k = alpha_l
    return alpha_k


def score_logistic(testX, testY, u, u_0, digit1, digit2):
    
    '''
    Tests the outputs obtained for the logistic problem
    
    '''
    
    
    t = (u @ testX.T + u_0)
    prob_digit2 = np.exp(t)/(1+np.exp(t))
    predictions = [digit1 if ti<0 else digit2 for ti in t]
    testing_bool = predictions == testY
    pct_wrong = 1-np.sum(testing_bool)/len(testY)
    
    prediction_df = pd.DataFrame({"Prob. of Digit 2" : prob_digit2,
                                  "Prediction": predictions,
                                  "Real Value": testY,
                                  "Test Result": testing_bool})
    
    return pct_wrong, prediction_df