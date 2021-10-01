# -*- coding: utf-8 -*-
"""
ass2_main.py

Purpose:
    Code for the second EQRM I assignment, subdivide tasks in different functions
    
Version:
    1       First start

Date:
    2021/01/10

Author: 
    Maurits van den Oever and Connor Stevens

To do:
    - get_data_daily (3 years of two stocks, put in df, MSFT and KO for now)
    - get daily corr estimates using NW est for middle year, bandwidth=1yr
    - from the middle year, get intra day data of one month, 5 min freq
    - get daily corr measures... and adapt frequency after to check result changes
"""

###########################################################
### Imports
import numpy as np
#import os
import pandas as pd
#from pandas_datareader import data as pdr
#from datetime import date
import yfinance as yf
#import scipy.optimize as opt
#import scipy.stats as st
#import quandl
#yf.pdr_override()


###########################################################
### get_data_daily
def get_data_daily(lTicks, sStart, sEnd):
    """
    Purpose:
        Gets daily data for specified tickers and dates
        
    Inputs:
        lTicks      List of strings of tickers
        sStart      string, start date of data
        sEnd        string, end date of data
    
    Output:
        dataframe with returns for both stocks
    """
    
    vMSFTClose = yf.Ticker(lTicks[0]).history(start = sStart, end = sEnd)['Close']
    vKOClose = yf.Ticker(lTicks[1]).history(start = sStart, end = sEnd)['Close']
    
    vMSFTRet = 100*(np.log(vMSFTClose) - np.log(vMSFTClose.shift(1)))
    vKORet = 100*(np.log(vKOClose) - np.log(vKOClose.shift(1)))
    
    dfret = pd.DataFrame({'vMSFTRet': vMSFTRet, 'vKORet': vKORet})
    dfret = dfret.iloc[1:,:]
    
    return dfret
    

    
    
    



###########################################################
### get_data_hf



###########################################################
### main

def main():
    # magic numbers and other variables specification:
    lTicks = ['MSFT', 'KO']
    sStart = '2018-01-01'
    sEnd = '2021-01-01'
    
    dfret_daily = get_data_daily(lTicks, sStart, sEnd)
    
    return  







###########################################################
### start main
if __name__ == "__main__":
    main()