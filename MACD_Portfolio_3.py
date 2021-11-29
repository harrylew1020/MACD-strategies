import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas_datareader as pdr
import datetime


def macdPortfolio_3(symbol1, symbol2, symbol3, start, end, a=12, b=26, c=9):
    price1 = pdr.get_data_yahoo(symbol1, start, end).drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], 1)
    price2 = pdr.get_data_yahoo(symbol2, start, end).drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], 1)
    price3 = pdr.get_data_yahoo(symbol3, start, end).drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], 1)
    years = (end - start).days / 365.25
    df = pd.DataFrame()
    df['Close1'] = price1
    df['Close2'] = price2
    df['Close3'] = price3
    df['Return1'] = price1.Close / price1.Close.shift(1)
    df['Return2'] = price2.Close / price2.Close.shift(1)
    df['Return3'] = price3.Close / price3.Close.shift(1)
    df.Return1.iloc[0] = 1
    df.Return2.iloc[0] = 1
    df.Return3.iloc[0] = 1
    starting_bal = 10000
    w = 1 / 3
    df['Bench_bal'] = starting_bal * ((w * df.Return1 + w * df.Return2 + w * df.Return3).cumprod())
    df['Bench_peak'] = df.Bench_bal.cummax()
    df['Bench_dd'] = (df.Bench_bal - df.Bench_peak) / df.Bench_peak
    Bench_maxDD = round(df.Bench_dd.min() * 100, 2)
    Bench_TotalReturn = round(((df.Bench_bal[-1] - df.Bench_bal[0]) / df.Bench_bal[-1]) * 100, 2)
    Bench_CAGR = round(((df.Bench_bal[-1] / df.Bench_bal[0]) ** (1 / years) - 1) * 100, 2)

    # Asset 1
    exp1_1 = df.Close1.ewm(span=a, adjust=False).mean()
    exp2_1 = df.Close1.ewm(span=b, adjust=False).mean()
    MACD1 = exp1_1 - exp2_1
    Signal1 = MACD1.ewm(span=c, adjust=False).mean()
    df['Long1'] = MACD1 > Signal1

    # Asset 2
    exp1_2 = df.Close2.ewm(span=a, adjust=False).mean()
    exp2_2 = df.Close2.ewm(span=b, adjust=False).mean()
    MACD2 = exp1_2 - exp2_2
    Signal2 = MACD2.ewm(span=c, adjust=False).mean()
    df['Long2'] = MACD2 > Signal2

    # Asset 3
    exp1_3 = df.Close3.ewm(span=a, adjust=False).mean()
    exp2_3 = df.Close3.ewm(span=b, adjust=False).mean()
    MACD3 = exp1_3 - exp2_3
    Signal3 = MACD3.ewm(span=c, adjust=False).mean()
    df['Long3'] = MACD3 > Signal3

    df['Sys_return1'] = np.where(df.Long1.shift(1) == True, df.Return1, 1)
    df['Sys_return2'] = np.where(df.Long2.shift(1) == True, df.Return2, 1)
    df['Sys_return3'] = np.where(df.Long3.shift(1) == True, df.Return3, 1)
    df['Sys_return'] = w * df.Sys_return1 + w * df.Sys_return2 + w * df.Sys_return3
    df['Sys_bal'] = starting_bal * df.Sys_return.cumprod()
    df['Sys_peak'] = df.Sys_bal.cummax()
    df['Sys_dd'] = (df.Sys_bal - df.Sys_peak) / df.Sys_peak

    Sys_maxDD = round(df.Sys_dd.min() * 100, 2)
    Sys_TotalReturn = round(((df.Sys_bal[-1] - df.Sys_bal[0]) / df.Sys_bal[-1]) * 100, 2)
    Sys_CAGR = round(((df.Sys_bal[-1] / df.Sys_bal[0]) ** (1 / years) - 1) * 100, 2)

    dframe = {'index title': ['Benchmark', 'System'], 'Total Return %': [Bench_TotalReturn, Sys_TotalReturn],
              'CAGR %': [Bench_CAGR, Sys_CAGR],
              'Maximum Drawdown %': [Bench_maxDD, Sys_maxDD]}
    backtest = pd.DataFrame(dframe).set_index('index title')

    fig = plt.figure()
    plt.subplot(211)
    plt.plot(df.Bench_bal, label='Benchmark')
    plt.plot(df.Sys_bal, label='System')
    plt.legend()
    plt.title(label=f'backtesting period: {start} to {end}')
    plt.subplot(212)
    plt.plot(MACD1, label='MACD')
    plt.plot(Signal1, label='Signal')
    plt.legend()

    return backtest, fig


symbol1='AAPL'
symbol2='BAC'
symbol3='NKE'
start=datetime.date(2006,1,1)
end=datetime.date(2010,1,1)

backtest, fig = macdPortfolio_3(symbol1, symbol2, symbol3, start, end, a=5, b=22, c=20)
#%%
plt.show()