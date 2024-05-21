import time
import datetime
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import data_process as d
import pandas_datareader.data as web 

get_ipython().run_line_magic('matplotlib', 'inline')
def initialize_q_mat(all_states, all_actions):
    #Creating Q-table
    states_size = len(all_states)
    actions_size = len(all_actions)
    
    q_mat = np.random.rand(states_size, actions_size)/1e9
    q_mat = pd.DataFrame(q_mat, columns=all_actions.keys())
    
    q_mat['states'] = all_states
    q_mat.set_index('states', inplace=True)
    
    return q_mat

def act(state, q_mat, threshold=0.2, actions_size=3):
    #Action function
    if np.random.uniform(0,1) < threshold: # go random
        action = np.random.randint(low=0, high=actions_size)  
    else:
        action = np.argmax(q_mat.loc[state].values)
    return action

def get_return_since_entry(bought_history, current_adj_close):
    #Return
    return_since_entry = 0.
    
    for b in bought_history:
        return_since_entry += (current_adj_close - b)
    return return_since_entry

def visualize_results(actions_history, returns_since_entry):
    #Visualize results
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,12))
    
    ax1.plot(returns_since_entry)
    
    days, prices, actions = [], [], []
    for d, p, a in actions_history:
        days.append(d)
        prices.append(p)
        actions.append(a)

    ax2.plot(days, prices, label='normalized adj close price')
    hold_d, hold_p, buy_d, buy_p, sell_d, sell_p = [], [], [], [], [], []
    for d, p, a in actions_history:
        if a == 0:
            hold_d.append(d)
            hold_p.append(p)
        if a == 1:
            buy_d.append(d)
            buy_p.append(p)
        if a == 2:
            sell_d.append(d)
            sell_p.append(p)
        
    ax2.scatter(hold_d, hold_p, color='blue', label='hold')
    ax2.scatter(buy_d, buy_p, color='green', label='buy')
    ax2.scatter(sell_d, sell_p, color='red', label='sell')
    ax2.legend()
    
def get_invested_capital(actions_history, returns_since_entry):
    #Get capital
    invest = []
    total = 0
    return_invest_ratio = None
    for i in range(len(actions_history)):
        a = actions_history[i][2]
        p = actions_history[i][1]

        try:
            next_a = actions_history[i+1][2]
        except:
            #print('end')
            break
        if a == 1:
            total += p
            #print(total)
            if next_a != 1 or (i==len(actions_history)-2 and next_a==1):
                invest.append(total)
                total = 0
    if invest:
        return_invest_ratio = returns_since_entry[-1]/max(invest)
        print('invested capital {}, return/invest ratio {}'.format(max(invest), return_invest_ratio))
    else:
        print('no buy transactions, invalid training')
    return return_invest_ratio
        
def get_base_return(data):
    #Benchmark return(Bazovaja pribilnost)
    
    start_price, _ = data[0]
    end_price, _ = data[-1]
    return (end_price - start_price)/start_price

start = datetime.datetime(2014, 1, 1)
end = datetime.datetime(2018, 1, 1)
train_df, test_df = d.get_stock_data('AAPL', start, end, 0.8)


train_df.head()


test_df = d.create_df(test_df, 3)
test_df = d.create_state_df(test_df, price_states_value, bb_states_value, close_sma_ratio_states_value)

all_actions = {0:'hold', 1:'buy', 2:'sell'}

train_df = d.create_df(train_df, 3)
price_states_value, bb_states_value, close_sma_ratio_states_value = d.get_states(train_df)
train_df = d.create_state_df(train_df, price_states_value, bb_states_value, close_sma_ratio_states_value)

all_states = d.get_all_states(price_states_value, bb_states_value, close_sma_ratio_states_value)
states_size = len(all_states)

def train_q_learning(train_data, q, alpha, gamma, episodes):
    #training Q-table
    actions_history = []
    num_shares = 0
    bought_history = []
    returns_since_entry = [0]
    for ii in range(episodes):
        actions_history = []
        num_shares = 0
        bought_history = []
        returns_since_entry = [0]
        days=[0]
        for i, val in enumerate(train_data):
            current_adj_close, state = val
            try:
                next_adj_close, next_state = train_data[i+1]
            except:
                break

            if len(bought_history) > 0:
                returns_since_entry.append(get_return_since_entry(bought_history, current_adj_close)) 
            else:
                returns_since_entry.append(returns_since_entry[-1])

            # decide action
            if alpha > 0.1:
                alpha = alpha/(i+1)
            action = act(state, q, threshold=alpha, actions_size=3)

            # get reward
            if action == 0: # hold
                if num_shares > 0:
                    prev_adj_close, _ = train_data[i-1]
                    future = next_adj_close - current_adj_close 
                    past = current_adj_close - prev_adj_close
                    reward = past
                else:
                    reward = 0

            if action == 1: # buy
                reward = 0
                num_shares += 1
                bought_history.append((current_adj_close))       

            if action == 2: # sell
                if num_shares > 0:
                    bought_price = bought_history[0]
                    reward = (current_adj_close - bought_price)
                    bought_history.pop(0)
                    num_shares -= 1

                else:
                    reward = -100
            actions_history.append((i, current_adj_close, action))
            
            # update q table
            q.loc[state, action] = (1.-alpha)*q.loc[state, action] + alpha*(reward+gamma*(q.loc[next_state].max()))
    print('End of Training!')
    return q, actions_history, returns_since_entry


def eval_q_learning(test_data, q):
    #reviewing Q-table
    actions_history = []
    num_shares = 0
    returns_since_entry = [0]
    bought_history = []
    
    for i, val in enumerate(test_data):
        current_adj_close, state = val
        try:
            next_adj_close, next_state = test_data[i+1]
        except:
            print('End of data! Done!')
            break   

        if len(bought_history) > 0:
            returns_since_entry.append(get_return_since_entry(bought_history, current_adj_close)) 
        else:
            returns_since_entry.append(returns_since_entry[-1])

        # decide action
        action = act(state, q, threshold=0, actions_size=3)

        if action == 1: # buy
            num_shares += 1
            bought_history.append((current_adj_close))
        if action == 2: # sell
            if num_shares > 0:
                bought_price = bought_history[0]
                bought_history.pop(0)
                num_shares -= 1

        actions_history.append((i, current_adj_close, action))

    return actions_history, returns_since_entry


train_df.head()



np.random.seed(12)
q_init = initialize_q_mat(train_df['norm_adj_close_state'].unique(), all_actions)
print('Initializing q')
print(q_init)


train_data = np.array(train_df[['norm_adj_close', 'norm_adj_close_state']])
q, train_actions_history, train_returns_since_entry = train_q_learning(train_data, q_init, alpha=0.8, gamma=0.95, episodes=1)

visualize_results(train_actions_history, train_returns_since_entry)
get_invested_capital(train_actions_history, train_returns_since_entry)
print('base return/invest ratio {}'.format(get_base_return(train_data)))


# ## test data


test_data = np.array(test_df[['norm_adj_close', 'norm_adj_close_state']])
test_actions_history, test_returns_since_entry = eval_q_learning(test_data, q)

visualize_results(test_actions_history, test_returns_since_entry)
get_invested_capital(test_actions_history, test_returns_since_entry)
print('base return/invest ratio {}'.format(get_base_return(test_data)))


# # improving model

np.random.seed(12)
q = initialize_q_mat(all_states, all_actions)/1e9
print('Initializing q')
print(q[:3])

train_data = np.array(train_df[['norm_adj_close', 'state']])
q, train_actions_history, train_returns_since_entry = train_q_learning(train_data, q, alpha=0.8, gamma=0.95, episodes=1)

visualize_results(train_actions_history, train_returns_since_entry)
get_invested_capital(train_actions_history, train_returns_since_entry)
print('base return/invest ratio {}'.format(get_base_return(train_data)))

test_data = np.array(test_df[['norm_adj_close', 'state']])
test_actions_history, test_returns_since_entry = eval_q_learning(test_data, q)

visualize_results(test_actions_history, test_returns_since_entry)
get_invested_capital(test_actions_history, test_returns_since_entry)
# print('invested capital {}, return/invest ratio {}'.format(invested_capital, return_invest_ratio))
print('base return/invest ratio {}'.format(get_base_return(test_data)))

