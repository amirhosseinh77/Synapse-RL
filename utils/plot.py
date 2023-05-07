import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

def plot_return(returns, window=10):
    display.clear_output(wait=True)
    plt.title('Score/Episode Plot')

    rolling_mean = pd.Series(returns).rolling(window).mean()
    std = pd.Series(returns).rolling(window).std()

    plt.plot(returns)
    plt.plot(rolling_mean)
    plt.fill_between(range(len(returns)),rolling_mean-std, rolling_mean+std, color='violet', alpha=0.2)
    plt.grid()
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Rolling Mean'])
    plt.pause(0.001)