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
    
    
To do:
    - make returns excess returns
    - get libor rates daily (not on yahoo finance for some f#@% reason)
    - redefine dS2/dNu in vLL function (figure out how to do it)
    - get function that optimizes vLL function
    - get function that gets covariance matrices...

Then we're pretty much done :D                        
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
### Data Puller
def Data_Puller(lTickers, sPath, sStart_date, sEnd_date):
    """
    Purpose:
        Pull data from Yahoo finance, having specified the needed tickers, start and end dates, 
        and filesPath for the data files

    Inputs:
        lTickers        list of strings, ticker names
        sPath           string of filepath for the data files
        sStart_date     string, start date in format yyyy-mm-dd
        sEnd_date       string, end date in format yyyy-mm-dd

    Author:
        Maurits van den Oever
    """
    
    
    lFiles = []
    
    # check if function has run/downloaded stuff before:
    if 'data_main.csv' in os.listdir(sPath):
        df = pd.read_csv(sPath+"data_main.csv",index_col=0)
        
    else:
        def SaveData(df, filename):
            df.to_csv(sPath +filename+".csv")
            
        
        def getData(ticker):
            print(ticker)
            data = yf.download(ticker, start=sStart_date, end=sEnd_date)
            dataname = ticker
            lFiles.append(dataname)
            SaveData(data, dataname)
            
        for tik in lTickers:
            getData(tik)
        
        
        df = pd.read_csv(sPath+str(lFiles[0])+".csv")
        df[str(lFiles[0])] = df['Adj Close']
        # filter df on adjclose and date:
        df = df.iloc[:,list([0,-1])]
        
        for i in range(1, len(lFiles)):
        #for i in range(1, 3):
            df1 = pd.read_csv(sPath+str(lFiles[i])+".csv")
            df1[str(lFiles[i])] = df1['Adj Close']
            df1 = df1.iloc[:,list([0,-1])]
            
            # now join those df1s to df for master dataset to get 
            df = pd.merge(df, df1, how='left', on=['Date'])
        
        # clean it up a bit, remove nans by ffill
        df = df.iloc[1:,:]
        df = df.ffill(axis=0)
    
        # get log returns for every ticker
        
        for tic in df.columns[1:]:
            df[tic+'_ret'] = np.log(df[tic]) - np.log(df[tic].shift(1))
            
        # get some portfolio returns, assume average weight...
        df['port_ret'] = df.iloc[:,len(lTickers)+1:len(df.columns)+1].mean(axis=1)
        df.to_csv(sPath+'data_main.csv')
    
    dfrets = df.iloc[1:,len(lTickers)+1:len(df.columns)-1]
    return df, dfrets

###########################################################
### dataloader function
def YahooFREDPull(sStart_date, sEnd_date, sTicker):
    """
    Purpose:
        Pull stock price data of a specified ticker and S&P500 from Yahoo Finance.
        
    Inputs:
        sStart_date     Date as string, start date of the stock prices
        sEnd_date       Date as string, end date of the stock prices
        sTicker         String, ticker of the stock
        
    Returns:
        
    """

    # get prices of all series: stock, s&p, and tbill3m
    vStockClose = yf.Ticker(sTicker).history(start = sStart_date, end = sEnd_date)['Close']
    vSPClose = yf.Ticker('^GSPC').history(start = sStart_date, end = sEnd_date)['Close']
    vTbillClose = quandl.get('FRED/DTB3', start_date = sStart_date, end_date = sEnd_date)
    
    vStockRet = 100*(np.log(vStockClose) - np.log(vStockClose.shift(1)))
    vSPRet = 100*(np.log(vSPClose) - np.log(vSPClose.shift(1)))
    vTbillClose *= (1/250)
    
    dfrets = pd.DataFrame({'vStockRet': vStockRet, 
                           'vSPRet': vSPRet, 
                           'vTbill': vTbillClose.iloc[:,0]})
    dfrets = dfrets.dropna(axis=0)
    dfrets['vStockRet'] = dfrets['vStockRet'] - dfrets['vTbill']
    dfrets['vSPRet'] = dfrets['vSPRet'] - dfrets['vTbill']
    
    dfrets = dfrets[list(['vStockRet', 'vSPRet'])]
    # write a csv real quick to load if needed:
    dfrets.to_csv(r'C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\data_main.csv')
    
    
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
    
    return vBeta

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
        dS2 = np.std(vE)
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
### main

def main():
    # define variables needed
    lTickers = ['NOW', '^GSPC'] # ri: service now, market: s&p, rf: 3m libor, ignore rf for now
    sTicker = 'KO'
    sPath = r"C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\\"
    sStart_date = '2000-09-15'
    sEnd_date = '2021-09-15'
    
    # define dataframe used
    dfrets = Data_Puller(lTickers, sPath, sStart_date, sEnd_date)[1] # (rets need some redefinition)
    vY, mX = matrix_definer_5003(dfrets) #define x and y matrices and vectors
    
    # calc beta from OLS given
    vBeta = CAPM_OLS(vY, mX)
    print('OLS betas are equal to:')
    print(vBeta)
    print('')
    
    # try to get some vLL given dists:
    vLLnorm = get_vLL(vY, mX, vBeta, 'normal')
    vLLt = get_vLL(vY, mX, vBeta, 't') # okay it works, not yet perfect though
    
    # vBetaML = 
    
    return  







###########################################################
### start main
if __name__ == "__main__":
    main()



