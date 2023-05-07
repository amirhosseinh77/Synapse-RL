import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

plt.style.use('fivethirtyeight')
plt.rc('font', size=10)

def plot_return(returns, agent, window=100):
    display.clear_output(wait=True)
    plt.figure(figsize=(8,4))
    plt.title(f'SYNAPS : {agent}')

    rolling_mean = pd.Series(returns).rolling(window).mean()
    std = pd.Series(returns).rolling(window).std()

    plt.plot(returns)
    plt.plot(rolling_mean)
    plt.fill_between(range(len(returns)),rolling_mean-std, rolling_mean+std, color='violet', alpha=0.4)
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Rolling Mean'])
    plt.pause(0.001)