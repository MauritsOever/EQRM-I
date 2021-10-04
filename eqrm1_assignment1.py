
'''
eqrm1_assignment1.py

Purpose:
    Compare OLS and ML methods of estimating the Beta parameter of the CAPM model.

Version:
    1       First start

Date:
    2021/09/18

Author:
    Connor Stevens and Maurits van den Oever
'''
###########################################################
### Imports
import numpy as np
import yfinance as yf
import quandl
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import scipy.optimize as opt

###########################################################
### (vStock, vMkt, vRfr) = YahooPull(datStart, datEnd, sTicker)
def YahooFREDPull(datStart, datEnd, sTicker):
    """
    Purpose:
    -------
        Pull stock price data of a specified ticker and S&P500 from Yahoo Finance. Also divides annual t-bill rate by 252 for daily rate.

    Inputs:
    ------
        datStart : Date, start date of stock prices.
        datEnd : Date, end date of stock prices.
        sTicker : String, ticker of the stock.

    Return value:
    ------------
        vTickclose : Vector, adjusted close for ticker for period.
        vSP500Close : Vector, S&P 500 closing price for period.
        vTbillClose : 3-month treasury bill rate for specified period, daily frequency.
    """

    vTickClose = yf.Ticker(sTicker) .history(start = datStart, end = datEnd)['Close']
    vSP500Close = yf.Ticker("^GSPC") .history(start = datStart, end = datEnd)['Close']
    vTbillClose = quandl.get("FRED/DTB3", start_date = datStart, end_date = datEnd)/252
    
    return vTickClose, vSP500Close, vTbillClose

###########################################################
### 
def RetCalc(vPriceSeries):
    """
    Purpose:
    -------
        Convert daily prices into log prices and then returns series.

    Inputs:
    ------
        vPriceSeries : Vector, series of specified interval price data.
    
    Return value:
    ------------
        vReturnSeries : Vector, adjusted close for time period.
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
    -------
        Create a single array out of the three inputs with matching dates using pandas. Also adds constant column to dataframe.

    Inputs:
    ------
        vRetStock : Vector, series of returns data.
        vRetMkt : Vector, series of market returns
        vRfr : Vector, series of risk-free-rate
    
    Return value:
    ------------
        vExRetSeries : Vector, adjusted close for time period.
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
    -------
        Calulate stock excess return by subtracting Rfr from stock return.

    Inputs:
    ------
        mData : Matrix(Dataframe), contains column with names 'stock_ret' and 'Rfr'.
    
    Return value:
    ------------
        mDataEx : Matrix(Dataframe), contains input columns and additional column of excess stock returns.
    """
    mData['stock_exRet'] = mData['stock_ret'] - mData['Rfr']
    
    mData['mkt_exRet'] = mData['mkt_ret'] - mData['Rfr']
    
    return mData

###########################################################
###
def XyExtract(mData):
    """
    Purpose:
    -------
        Output X matrix and y vector for easy linear algebra computation.

    Inputs:
    ------
        mData : Matrix(Dataframe), contains column with names 'stock_ret', 'mkt_ret(X)', 'Rfr', 'stock_exRet' and 'constant'.
    
    Return values:
    -------------
        mX : Matrix(Dataframe), contains column of market returns and column of constants.
        vy : Vector, contains stock excess returns.
    """
    mX = mData[['constant', 'mkt_exRet']]
    vy = mData['stock_exRet']
    
    return mX, vy

###########################################################
### res.x= EstRegNorm(vY, mX)
def EstRegNorm(vY, mX):
    """
    Purpose:
    -------
        Estimate the regression model using ML assuming normal distribution.

    Inputs:
    ------
        vY : iN vector of observations
        mX : iN x iK matrix of explanatory variables

    Return value:
    ------------
        vP : iK+1 vectors of sigma and beta's, estimated
    """
    (iN, iK)= mX.shape
    vP0= np.ones(iK+1)       # Bad starting values
    AvgNLnLReg= lambda vP: -np.mean(LnLRegNorm(vP, vY, mX), axis=0)
    

    print ('Initial LLnormal= {}'.format(-iN*AvgNLnLReg(vP0)))

    res= opt.minimize(AvgNLnLReg, vP0, method='BFGS')
    print ('\nResults_normal: ', res)
    print("\ndLL_normal=", -iN*res.fun)
    return res.x

###########################################################
### vLL= LnLRegNorm(vP, vY, mX)
def LnLRegNorm(vP, vY, mX):
    """
    Purpose:
    -------
        Calculate vector of LL

    Inputs:
    ------
        vP : iK+1 vectors of sigma (sigma always vP[0]) and beta's.
        vY : iN vector of realised returns on the stock.
        mX : iN x iK matrix of excess market returns with a the first column a consta

    Return value:
    ------------
        vLL : iN vector of loglikelihoods
    """
    
    (dS, vBeta)= (vP[0], vP[1:])
    vE= vY - mX@vBeta
    vLL= st.norm.logpdf(vE , scale = dS)
    print ('.', end='')
    return vLL

###########################################################
### res.x= EstRegT(vY, mX)
def EstRegT(vY, mX):
    """
    Purpose:
    -------
    Estimate the regression model using ML assuming Student-t distribution.

    Inputs:
    ------
        vY : iN vector of observations
        mX : iN x iK matrix of explanatory variables

    Return value:
    ------------
        vP : iK+1 vectors of estimated (optimized) sigma[0], df[1] and betas[2:].
    """
    (iN, iK)= mX.shape
    vP0= np.ones(iK+2)       # Bad starting values
    AvgNLnLReg= lambda vP: -np.mean(LnLRegT(vP, vY, mX), axis=0)
    

    print ('\n\n\nInitial LLt= {}'.format(-iN*AvgNLnLReg(vP0)))

    res= opt.minimize(AvgNLnLReg, vP0, method='BFGS')
    print ('\nResults_t: ', res)
    print("\ndLL_t=", -iN*res.fun)
    return res.x

###########################################################
### vLLt= LnLRegT(vP, vY, mX)
def LnLRegT(vP, vY, mX):
    """
    Purpose:
        Compute a vector of loglikelihoods of the regression model using tdist

    Inputs:
        vP      iK+1 vectors of sigma and beta's
        vY      iN vector of observations
        mX      iN x iK matrix of explanatory variables

    Return value:
        vLLt     iN vector of loglikelihoods
    """
    
    (dS, df, vBeta)= (vP[0], vP[1], vP[2:])
    vE= vY - mX@vBeta
    vLLt= st.t. logpdf (vE , df, scale = dS)
    print ('.', end='')
    return vLLt

###########################################################
### mH= hessian_2sided(fun, vP, *args)
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
### vG= gradient_2sided(fun, vP, *args)
def gradient_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical gradient, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, as used for minimize()
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      vG      iP vector with gradient

    See also:
      scipy.optimize.approx_fprime, for forward difference
    """
    iP = np.size(vP)
    vP= np.array(vP).reshape(iP)      # Ensure vP is 1D-array

    # f = fun(vP, *args)    # central function value is not needed
    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    fp = np.zeros(iP)
    fm = np.zeros(iP)
    for i in range(iP):     # Find f(x+h), f(x-h)
        fp[i] = fun(vP+mh[i], *args)
        fm[i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    vG= (fp - fm) / (vhr + vhl)  # Get central gradient

    return vG

###########################################################
### mG= jacobian_2sided(fun, vP, *args)
def jacobian_2sided(fun, vP, *args):
    """
    Purpose:
      Compute numerical jacobian, using a 2-sided numerical difference

    Author:
      Charles Bos, following Kevin Sheppard's hessian_2sided, with
      ideas/constants from Jurgen Doornik's Num1Derivative

    Inputs:
      fun     function, return 1D array of size iN
      vP      1D array of size iP of optimal parameters
      args    (optional) extra arguments

    Return value:
      mG      iN x iP matrix with jacobian

    See also:
      numdifftools.Jacobian(), for similar output
    """
    iP = np.size(vP)
    vP= vP.reshape(iP)      # Ensure vP is 1D-array

    vF = fun(vP, *args)     # evaluate function, only to get size
    iN= vF.size

    vh= _gh_stepsize(vP)
    mh = np.diag(vh)        # Build a diagonal matrix out of h

    mGp = np.zeros((iN, iP))
    mGm = np.zeros((iN, iP))

    for i in range(iP):     # Find f(x+h), f(x-h)
        mGp[:,i] = fun(vP+mh[i], *args)
        mGm[:,i] = fun(vP-mh[i], *args)

    vhr = (vP + vh) - vP    # Check for effective stepsize right
    vhl = vP - (vP - vh)    # Check for effective stepsize left
    mG= (mGp - mGm) / (vhr + vhl)  # Get central jacobian

    return mG

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
###########################################################
### main
def main():
    # Magic numbers
    sTicker = "KO"
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
    (mX, vY) = XyExtract(mData)
    
    #OLS
    vBeta_hat = np.linalg.inv(np.transpose(mX) @ mX) @ np.transpose(mX) @ vY
    
    #ML_norm
    vP_normal = EstRegNorm(vY, mX)
    
    #Covariance matrix normal dist.
    mG_normal= jacobian_2sided (LnLRegNorm , vP_normal , vY , mX)
    mBopg_normal= mG_normal.T@mG_normal/len(mX)
    AvgNLnLReg= lambda vP_normal: -np.mean(LnLRegNorm(vP_normal, vY, mX), axis=0)
    mH_normal = hessian_2sided(AvgNLnLReg, vP_normal)
    mA0Ni_normal= -np.linalg.inv(mH_normal)
    mS2a_normal= mA0Ni_normal*mBopg_normal*mA0Ni_normal
    print("\n\nmS2a_normal covariance matrix= ", mS2a_normal)
    
    #Standard errors.
    dSE_sig_n = np.sqrt(mS2a_normal[0, 0])/np.sqrt(len(mX))
    dSE_B0_n = np.sqrt(mS2a_normal[1, 1])/np.sqrt(len(mX))
    dSE_B1_n = np.sqrt(mS2a_normal[2, 2])/np.sqrt(len(mX))
    print('SE {}, {}, {}'.format(dSE_sig_n, dSE_B0_n, dSE_B1_n))
    
    #t-stats.
    dt_sig_n = vP_normal[0]/dSE_sig_n
    dt_B0_n =vP_normal[1]/dSE_B0_n
    dt_B1_n =vP_normal[2]/dSE_B1_n
    print('t-stats {}, {}, {}'.format(dt_sig_n, dt_B0_n, dt_B1_n))
    
    #--------------------------
    
    #ML_t
    vP_t = EstRegT(vY, mX)
    
    #Covariance matrix t dist.
    mG_t= jacobian_2sided (LnLRegT , vP_t , vY , mX)
    mBopg_t= mG_t.T@mG_t/len(mX) 
    AvgNLnLReg= lambda vP_t: -np.mean(LnLRegT(vP_t, vY, mX), axis=0)
    mH_t = hessian_2sided(AvgNLnLReg, vP_t)
    mIh_t= -mH_t
    mA0Ni_t= -np.linalg.inv(mH_t)
    mS2a_t= mA0Ni_t*mBopg_t*mA0Ni_t
    print("\n\nmS2a_t covariance matrix= ", mS2a_t)
    
    
    #Standard errors.
    dSE_sig_t = np.sqrt(mS2a_t[0, 0])/np.sqrt(len(mX))
    dSE_df_t = np.sqrt(mS2a_t[1, 1])/np.sqrt(len(mX))
    dSE_B0_t = np.sqrt(mS2a_t[2, 2])/np.sqrt(len(mX))
    dSE_B1_t = np.sqrt(mS2a_t[3, 3])/np.sqrt(len(mX))
    print('SE {}, {}, {}, {}'.format(dSE_sig_t,dSE_df_t, dSE_B0_t, dSE_B1_t))
    
    #t-stats.
    dt_sig_t = vP_t[0]/dSE_sig_t
    dt_df_t = vP_t[1]/dSE_df_t
    dt_B0_t =vP_t[2]/dSE_B0_t
    dt_B1_t =vP_t[3]/dSE_B1_t
    print('t-stats {}, {}, {}, {'.format(dt_sig_t,dt_df_t, dt_B0_t, dt_B1_t))
    
    

###########################################################
### start main
if __name__ == "__main__":
    main()
