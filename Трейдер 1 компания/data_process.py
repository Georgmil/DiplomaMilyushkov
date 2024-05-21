import datetime
import numpy as np
import pandas  as pd
import pandas_datareader.data as web 

def get_stock_data(symbol, start, end, train_size=0.8):

    #Get stock data in the given date range
    df = web.DataReader(symbol, 'yahoo', start, end)
    
    train_len = int(df.shape[0] * train_size)
    
    if train_len > 0:
        train_df = df.iloc[:train_len, :]
        test_df = df.iloc[train_len:, :]
        return train_df, test_df
    else:
        return df


def get_bollinger_bands(values, window):
    #Return upper and lower Bollinger Bands.
    
    rm = values.rolling(window=window).mean()
    rstd = values.rolling(window=window).std()
    
    band_width = 2. * rstd / rm
    return band_width.apply(lambda x: round(x,5))

def get_adj_close_sma_ratio(values, window):

    #Return the ratio of adjusted closing value to the simple moving average.

    rm = values.rolling(window=window).mean()
    ratio = values/rm
    return ratio.apply(lambda x: round(x,5))

def discretize(values, num_states=10):
    #Convert continuous values to integer state
   
    states_value = dict()
    step_size = 1./num_states
    for i in range(num_states):
        if i == num_states - 1:
            states_value[i] = values.max()
        else:
            states_value[i] = values.quantile((i+1)*step_size)
    return states_value

def value_to_state(value, states_value):
    #Convert values to state

    if np.isnan(value):
        return np.nan
    else:
        for state, v in states_value.items():
            if value <= v:
                return str(state)
        return 'value out of range'

def create_df(df, window=3):

    #Create a dataframe w
    
    # get bollinger value
    bb_width = get_bollinger_bands(df['Adj Close'], window)
    # get the ratio of close price to simple moving average
    close_sma_ratio = get_adj_close_sma_ratio(df['Close'], window)
    
    df['bb_width'] = bb_width
    df['close_sma_ratio'] = close_sma_ratio
    
    # drop missing values
    df.dropna(inplace=True)
    
    # normalize close price
    df['norm_adj_close'] = df['Adj Close']/df.iloc[0,:]['Adj Close']
    df['norm_bb_width'] = df['bb_width']/df.iloc[0,:]['bb_width']
    df['norm_close_sma_ratio'] = df['close_sma_ratio']/df.iloc[0,:]['close_sma_ratio']
    
    return df

def get_states(df):
    
    # discretize values
    price_states_value = discretize(df['norm_adj_close'])
    bb_states_value = discretize(df['norm_bb_width'])
    close_sma_ratio_states_value = discretize(df['norm_close_sma_ratio'])
    
    return price_states_value, bb_states_value, close_sma_ratio_states_value


def create_state_df(df, price_states_value, bb_states_value, close_sma_ratio_states_value):
    
    df['bb_width_state'] = df['bb_width'].apply(lambda x : value_to_state(x, bb_states_value))
    df['close_sma_ratio_state'] = df['close_sma_ratio'].apply(lambda x : value_to_state(x, close_sma_ratio_states_value))
    df['norm_adj_close_state'] = df['norm_adj_close'].apply(lambda x : value_to_state(x, price_states_value))
    
    df['state'] = df['norm_adj_close_state'] + df['close_sma_ratio_state'] + df['bb_width_state']
    df.dropna(inplace=True)
    return df

def get_all_states(price_states_value, bb_states_value, close_sma_ratio_states_value):

    states = []
    for p, _ in price_states_value.items():
        for c, _ in close_sma_ratio_states_value.items():
            for b, _ in bb_states_value.items():
                state =  str(p) + str(c) + str(b)
                states.append(str(state))
    
    return states

