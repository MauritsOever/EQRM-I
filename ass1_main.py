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
# import pandas as pd
# import matplotlib.pyplot as plt


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
    
    # specify packages just in case:
    import os
    import numpy as np
    import pandas as pd
    from pandas_datareader import data as pdr
    from datetime import date
    import yfinance as yf
    yf.pdr_override()
    
    # hardcode all the arguments bc one assignment anyways, next time i can automate more easily
    # main args
    lTickers = ['ASML.AS', 'SONY']
    #sPath = r"C:\Users\gebruiker\Documents\GitHub\QFRM\Data3\\"
    
    # some more args
    #sStart_date = '2011-04-20'
    #sEnd_date = '2021-04-20'
    lFiles = []
    
    # check if function has run/downloaded stuff before:
    if 'data_main.csv' in os.listdir(sPath):
        df = pd.read_csv(sPath+"data_main.csv",index_col=0)
        
    else:
        def SaveData(df, filename):
            df.to_csv(sPath +filename+".csv")
            
        
        def getData(ticker):
            print(ticker)
            data = pdr.get_data_yahoo(ticker, start=sStart_date, end=sEnd_date)
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
### main

def main():
    lTickers = ['ASML.AS', 'SONY']
    sPath = r"C:\Users\gebruiker\Documents\GitHub\EQRM-I\Data1\\"
    sStart_date = '2011-04-20'
    sEnd_date = '2021-04-20'
    
    
    df, dfrets = Data_Puller(lTickers, sPath, sStart_date, sEnd_date)
    
    return 







###########################################################
### start main
if __name__ == "__main__":
    main()


