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
import wrds


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
def get_data_hf(lTicks, sStartHF, sEndHF):
    
    return

###########################################################
### daily_corrs(dfret_daily)

def daily_corrs(dfret_daily, est_type, kernel_type):
    """
    Purpose:
        returns daily estimates of correlation coefficients between the two stocks
    
    Inputs:
        dfret_daily         df with daily returns
        est_type            string, estimation types, can choose between NW
        kernel_type         string, kernel type, can choose between ''
        
    To do:
        - get vector of weight based on kernel... not really needed, did the multiplication autimatically in the same line bc kernel func
        - get NW estimate for average, {{{DONE}}}
        - get variances based on rets 
        - get cov based on rets
        - calc rho for that t
        
        - loop over middle year
    """
    # magic numbers to screw around with
    dh = 0.5 # in years...
    start_date = pd.to_datetime('2019-01-02')
    end_date = pd.to_datetime('2019-12-31')
    # have the whole routine start at the first date of 2019
    
    
    dfret_daily['vMSFT_NW'] = np.full(len(dfret_daily), np.nan) # to store NW estimates, later used for variance and stuff
    dfret_daily['vKO_NW'] = np.full(len(dfret_daily), np.nan)
    
    for i in dfret_daily[start_date:end_date].index:
        numerator_msft = 0
        numerator_ko = 0
        denominator = 0
        for j in dfret_daily.index:
            if (i - j).days / 365 < dh:
                numerator_msft += 0.5*dfret_daily['vMSFTRet'][j]
                numerator_ko += 0.5*dfret_daily['vKORet'][j]
                denominator += 0.5
                
        dfret_daily['vMSFT_NW'][i] = numerator_msft / denominator
        dfret_daily['vKO_NW'][i] = numerator_ko / denominator
        
        print(dfret_daily['vMSFT_NW'][i]) # look at estimates, see if it makes sense
        
    # now get var, covar and then rho is ezpz

###########################################################
### main

def main():
    # magic numbers and other variables specification:
    lTicks = ['MSFT', 'KO']
    sStart = '2018-01-01'
    sEnd = '2021-01-01'
    
    sStartHF = '2019-01-01'
    sEndHF = '2019-02-01'
    
    dfret_daily = get_data_daily(lTicks, sStart, sEnd)
    
    return  







###########################################################
### start main
if __name__ == "__main__":
    main()