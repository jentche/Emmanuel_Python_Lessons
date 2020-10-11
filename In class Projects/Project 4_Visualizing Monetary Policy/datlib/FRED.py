#FRED.py
import pandas as pd
import pandas_datareader.data as web
import datetime

# creating a function that imports data from the federal reserve
# First, we create a dictionary of data codes, so each code is linked to a name
# frequency A = Annual, W = weekly, M = monthly, Q = quarterly, D = daily
# you can also add a number next to the frequency so, Freq = 2D means every 2 days, 3M means very 3 months, etc.
def gather_data(data_codes, start, end = datetime.datetime.today(), freq = "A"):
 
# check if new column is first column of data, 
# if true, then create a new df
    i = 0
    
# dct.items() calls key and value that key points to
# run through each code, key is column name and val is the code    
    for key, code in data_codes.items():
# check if it's our first column of data        
        if i == 0:
# if it's our first column of data, create a new df     
# Create dataframe for first variable, then rename column
# and resample the data by frequency and take the mean so if you sampled the data bi weekly, you'll take the mean or weekly average
# instead of .mean(), you can also use .first() or .last to get just the first sample, or last sample
            df = web.DataReader(code, "fred", start, end).resample(freq).mean()
            
# rename first column so that code is replaced by the key (variable name)            
            df.rename(columns = {code:key}, inplace = True) 

# setting i to None will cause the next block of code to execute,
# placing data within df instead of creating a new dataframe for each variable
            i = None
        else:
            
# If dataframe already exists, add new column
            df[key] = web.DataReader(code, "fred", start, end).resample(freq).mean()

    return df

# So, we'll pass a column of data to this function to convert the values from billion to million
def bil_to_mil(series):
# to convert, multiply the value in the series by a thousand    
    return series * 10**3