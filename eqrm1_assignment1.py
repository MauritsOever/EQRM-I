
'''
eqrm1_assignment1.py

Purpose:
    Compare OLS and ML methods of estimating the Beta parameter of the CAPM model.

Version:
    1       First start

Date:
    2021/09/18

Author:
    Connor Stevens
'''
###########################################################
### Imports
import numpy as np
import yfinance as yf
import quandl
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

###########################################################
### (vStock, vMkt, vRfr) = YahooPull(datStart, datEnd, sTicker)
def YahooFREDPull(datStart, datEnd, sTicker):
    """
    Purpose:
        Pull stock price data of a specified ticker and S&P500 from Yahoo Finance.

    Inputs:
        datStart        Date, start date of stock prices.
        datEnd          Date, end date of stock prices.
        sTicker         String, ticker of the stock.

    Return value:
        vTickclose       Vector, adjusted close for ticker for period.
        vSP500Close      Vector, S&P 500 closing price for period.
        vTbillClose      3-month treasury bill rate for specified period, daily frequency.
    """

    vTickClose = yf.Ticker(sTicker) .history(start = datStart, end = datEnd)['Close']
    vSP500Close = yf.Ticker("^GSPC") .history(start = datStart, end = datEnd)['Close']
    vTbillClose = quandl.get("FRED/DTB3", start_date = datStart, end_date = datEnd)
    
    return vTickClose, vSP500Close, vTbillClose

###########################################################
### 
def RetCalc(vPriceSeries):
    """
    Purpose:
        Convert daily prices into log prices and then returns series.

    Inputs:
        vPriceSeries       Vector, series of specified interval price data.
    
    Return value:
        vReturnSeries      Vector, adjusted close for time period.
    """
    
    #Log prices
    vPriceSeries = pd.DataFrame(np.log(vPriceSeries))
    #print(vPriceSeries.head())
    
    
    vReturnSeries = vPriceSeries.diff() * 100
    #print(vReturnSeries.head)
        
    return vReturnSeries

###########################################################
### 
def DateMatch(vRetStock, vRetMkt, vRfr):
    """
    Purpose:
        Create a single array out of the three inputs with matching dates using pandas. Also adds constant column to dataframe.

    Inputs:
        vRetStock       Vector, series of returns data.
        vRetMkt           Vector, series of market returns
        vRfr              Vector, series of risk-free-rate
    
    Return value:
        vExRetSeries      Vector, adjusted close for time period.
    """
    
    mDateMatch = pd.merge(pd.merge(vRetStock, vRetMkt, on='Date', how = 'inner'), vRfr,on='Date', how = 'inner')
    
    mDateMatch = mDateMatch.rename(columns={"Close_x": "stock_ret", "Close_y": "mkt_ret", "Value": "Rfr"})
    
    mDateMatch['constant'] = 1
    
    return mDateMatch

###########################################################
###
def ExcessRet(mData):
    """
    Purpose:
        Calulate stock excess return by subtracting Rfr from stock return.

    Inputs:
        mData       Matrix(Dataframe), contains column with names 'stock_ret' and 'Rfr'.
    
    Return value:
        mDataEx     Matrix(Dataframe), contains input columns and additional column of excess stock returns.
    """
    mData['stock_exRet'] = mData['stock_ret'] - mData['Rfr']
    
    return mData

###########################################################
###
def XyExtract(mData):
    """
    Purpose:
        Output X matrix and y vector for easy linear algebra computation.

    Inputs:
        mData       Matrix(Dataframe), contains column with names 'stock_ret', 'mkt_ret(X)', 'Rfr', 'stock_exRet' and 'constant'.
    
    Return values:
        mX     Matrix(Dataframe), contains column of market returns and column of constants.
        vy     Vector, contains stock excess returns.
    """
    mX = mData[['constant', 'mkt_ret']]
    vy = mData['stock_exRet']
    
    return mX, vy

###########################################################
### main
def main():
    # Magic numbers
    sTicker = "MSFT"
    datStart = "2000-01-01"
    datEnd = "2021-09-01"

    ## Initialisation
    #Pull data
    (vStock, vMkt, vRfr) = YahooFREDPull(datStart, datEnd, sTicker)

    #Prices to returns
    (vRetStock, vRetMkt) = RetCalc(vStock), RetCalc(vMkt)
    
    #Match dates
    mData = DateMatch(vRetStock, vRetMkt, vRfr)
    
    #Calc excess returns
    mData = ExcessRet(mData)
    
    #Get X and y
    (mX, vy) = XyExtract(mData)
    
    # Estimation
    vBeta_hat = np.linalg.inv(np.transpose(mX) @ mX) @ np.transpose(mX) @ vy
    # Output
    

###########################################################
### start main
if __name__ == "__main__":
    main()
