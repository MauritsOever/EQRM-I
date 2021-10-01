# -*- coding: utf-8 -*-
"""
ass1_main.py

Purpose:
    Code for the first EQRM I assignment, subdivide tasks in different functions
    
Version:
    1       First start

Date:
    2021/13/9

Author: 
    Maurits van den Oever                      
"""


###########################################################
### Imports
import numpy as np
import os
import pandas as pd
#from pandas_datareader import data as pdr
#from datetime import date
import yfinance as yf
import scipy.optimize as opt
import scipy.stats as st
import quandl
yf.pdr_override()


###########################################################
### YahooFREDPull
def YahooFREDPull(sStart_date, sEnd_date, sTicker):
    """
    Purpose:
        Pull stock price data of a specified ticker and S&P500 from Yahoo Finance.
        
    Inputs:
        sStart_date     Date as string, start date of the stock prices
        sEnd_date       Date as string, end date of the stock prices
        sTicker         String, ticker of the stock
        
    Returns:
        dataframe containing excess returns for S&P and the stock
    """

    # get prices of all series: stock, s&p, and tbill3m
    # vStockClose = yf.Ticker(sTicker).history(start = sStart_date, end = sEnd_date)['Close']
    # vSPClose = yf.Ticker('^GSPC').history(start = sStart_date, end = sEnd_date)['Close']
    # vTbillClose = quandl.get('FRED/DTB3', start_date = sStart_date, end_date = sEnd_date)
    
    # vStockRet = 100*(np.log(vStockClose) - np.log(vStockClose.shift(1)))
    # vSPRet = 100*(np.log(vSPClose) - np.log(vSPClose.shift(1)))
    # vTbillClose *= (1/250)
    
    # dfrets = pd.DataFrame({'vStockRet': vStockRet, 
    #                         'vSPRet': vSPRet, 
    #                         'vTbill': vTbillClose.iloc[:,0]})
    # dfrets = dfrets.dropna(axis=0)
    # dfrets['vStockRet'] = dfrets['vStockRet'] - dfrets['vTbill']
    # dfrets['vSPRet'] = dfrets['vSPRet'] - dfrets['vTbill']
    
    # dfrets = dfrets[list(['vStockRet', 'vSPRet'])]
    # # write a csv real quick to load if needed:
    # dfrets.to_csv(r'C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\data_main.csv')
    
    
    # load in instead of pull for quandl limits, but hand it in w/ code above cuz then its ready to run:
    dfrets = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\data_main.csv')
    
    
    return dfrets

###########################################################
### matrix definer 5003

def matrix_definer_5003(dfrets):
    """
    Purpose:
        define matrices for the CAPM model, namely the vector Y which holds and 
        the matrix X
        
    Inputs:
        dfrets, dataframe that holds returns
    
    Author:
        Maurits van den Oever
    """
    vY = np.array(dfrets['vSPRet'])
    mX = np.ones((len(dfrets),2))
    mX[:,1] = np.array(dfrets['vStockRet']) # these are not excess yet
    
    return vY, mX

###########################################################
### CAPM OLS estimator

def CAPM_OLS(vY, mX):
    """
    Purpose:
        Estimate beta for the standard OLS CAPM model using rm, ri and rf
    
    Inputs:
        vY
        mX
    
    Author:
        Maurits van den Oever
    """
    
    vBeta = np.linalg.inv(np.transpose(mX)@mX)@np.transpose(mX)@vY
    
    # standard errors...
    vEHat = vY - mX@vBeta
    dN = len(vY)
    dSigmaHat = (1/dN)*np.sum(vEHat**2)
    
    vSE_OLS = dSigmaHat * np.linalg.inv((np.transpose(mX)@mX))
    
    return vBeta, vSE_OLS

###########################################################
### vLLn/t
def get_vLL(vY, mX, vBeta, sDist):
    """
    Purpose:
        Returns a vector of log likelihoods given the inputs
    
    Inputs:
        vY      vector of Y data
        mX      X matrix
        vBeta   vector containing beta's
        sDist   string denoting which distribution to use
    """
    vYhat = np.matmul(mX, vBeta)
    vE = vY - vYhat
    
    # just to get a scalar for now, but dS2 needs to be redefined
    
    
    if sDist == 'normal':
        dS2 = np.std(vE)**2
        vLL = st.norm.logpdf(vE, scale= np.sqrt(dS2))
    elif sDist == 't':
        dNu, _, dS2 = st.t.fit(vE)
        vLL = st.t.logpdf(vE, df= dNu, scale= np.sqrt(dS2))
    else: 
        print('Please pick a supported distribution, either normal for normal or t for student t')
        vLL = 'error'
        
    return vLL

###########################################################
### ML optimizer
def vLL_optimizer(vBeta, vY, mX, sDist):
    """
    Purpose:
        optimizes the get_vLL function to return vBeta_hat ML
    
    Inputs:
        vBeta
        vY
        mX
        sDist
    """
    AvgNLL= lambda vBeta: -np .mean(get_vLL(vY, mX , vBeta, sDist))
    res   = opt.minimize(AvgNLL, vBeta, method="BFGS")
    print(res.message)
    print('vBetahat_ML = ', res.x)
    return res

###########################################################
### vh= _gh_stepsize(vP)
def _gh_stepsize(vP):    
    """    
    Purpose:        
        Calculate stepsize close (but not too close) to machine precision    
        
    Inputs:        
        vP      1D array of parameters    
    
    Return value:        
        vh      1D array of step sizes    
    """    
    vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize    
    vh= np.maximum(vh, 5e-6)       # Don't go too small    
    return vh


############################################################## mH= hessian_2sided(fun, vP, *args)
def hessian_2sided(fun, vP, *args):    
    """    
    Purpose:      
        Compute numerical hessian, using a 2-sided numerical difference    
        
    Author:      
        Kevin Sheppard, adapted by Charles Bos    
    
    Source:      
        https://www.kevinsheppard.com/Python_for_Econometrics  
        
    Inputs:      
        fun     function, as used for minimize()      
        vP      1D array of size iP of optimal parameters      
        args    (optional) extra arguments    
    
    Return value:      
        mH      iP x iP matrix with symmetric hessian    
    """    
    iP = np.size(vP,0)    
    vP= vP.reshape(iP)    # Ensure vP is 1D-array    
    f = fun(vP, *args)    
    vh= _gh_stepsize(vP)    
    vPh = vP + vh    
    vh = vPh - vP    
    mh = np.diag(vh)            # Build a diagonal matrix out of vh    
    fp = np.zeros(iP)    
    fm = np.zeros(iP)
    
    for i in range(iP):        
        fp[i] = fun(vP+mh[i], *args)        
        fm[i] = fun(vP-mh[i], *args)    
        
    fpp = np.zeros((iP,iP))    
    fmm = np.zeros((iP,iP))
    
    for i in range(iP):        
        for j in range(i,iP):            
            fpp[i,j] = fun(vP + mh[i] + mh[j], *args)            
            fpp[j,i] = fpp[i,j]            
            fmm[i,j] = fun(vP - mh[i] - mh[j], *args)            
            fmm[j,i] = fmm[i,j]    
            
    vh = vh.reshape((iP,1))   
    mhh = vh @ vh.T             # mhh= h h', outer product of h-vector
    mH = np.zeros((iP,iP))    
    for i in range(iP):        
        for j in range(i,iP):            
            mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2            
            mH[j,i] = mH[i,j]    
            
    return mH

###########################################################
### main

def main():
    # define variables needed
    sTicker = 'KO'
    # sPath = r"C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\\"
    sStart_date = '2000-09-15'
    sEnd_date = '2021-09-15'
    
    # define dataframe used
    dfrets = YahooFREDPull(sStart_date, sEnd_date, sTicker)
    vY, mX = matrix_definer_5003(dfrets) #define x and y matrices and vectors
    
    # calc beta from OLS given
    vBeta, vSE_OLS = CAPM_OLS(vY, mX)
    print('OLS betas are equal to:')
    print(vBeta)
    print('')
    print('OLS standard errors are: ')
    print(vSE_OLS)
    print('')
    
    # ML norm starts here
    resML_norm = vLL_optimizer(vBeta, vY, mX, 'normal')
    avgNLL_norm = lambda vBeta: -np .mean(get_vLL(vY, mX , vBeta, 'normal'))
    mHessian_norm = -hessian_2sided(avgNLL_norm, resML_norm.x) # seems correct...
    
    # get covariance matrix non robust
    
    # now robust
    
    # ML t starts here
    resML_t = vLL_optimizer(vBeta, vY, mX, 't') # nice estimates...
    avgNLL_t = lambda vBeta: -np.mean(get_vLL(vY, mX, vBeta, 't'))
    mHessian_t = -hessian_2sided(avgNLL_t, resML_t.x) # seems way too big
    
    
    
    return  







###########################################################
### start main
if __name__ == "__main__":
    main()



