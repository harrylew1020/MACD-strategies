import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pandas_datareader as pdr
import datetime


def macd(symbols, start, end, a=12, b=26, c=9, positiveMACD=True):
    # benchmark calculation
    years = (end - start).days / 365.25
    price = pdr.get_data_yahoo(symbols, start, end)
    price = price.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], 1)
    price['Return'] = price.Close / price.Close.shift(1)
    price.Return.iloc[0] = 1
    starting_bal = 10000
    price['Bench_bal'] = starting_bal * price.Return.cumprod()
    price['Bench_peak'] = price.Bench_bal.cummax()
    price['Bench_dd'] = (price.Bench_bal - price.Bench_peak) / price.Bench_peak
    Bench_maxDD = round(price.Bench_dd.min() * 100, 2)
    Bench_TotalReturn = round(((price.Bench_bal[-1] - price.Bench_bal[0]) / price.Bench_bal[-1]) * 100, 2)
    Bench_CAGR = round(((price.Bench_bal[-1] / price.Bench_bal[0]) ** (1 / years) - 1) * 100, 2)

    # MACD calculations
    exp1 = price.Close.ewm(span=a, adjust=False).mean()
    exp2 = price.Close.ewm(span=b, adjust=False).mean()
    MACD = exp1 - exp2
    Signal = MACD.ewm(span=c, adjust=False).mean()
    if positiveMACD == 'True':
        price['Long'] = (MACD > Signal) & (Signal > 0)
    else:
        price['Long'] = MACD > Signal

    price['Sys_return'] = np.where(price.Long.shift(1) == True, price.Return, 1)
    price['Sys_bal'] = starting_bal * price.Sys_return.cumprod()
    price['Sys_peak'] = price.Sys_bal.cummax()
    price['Sys_dd'] = (price.Sys_bal - price.Sys_peak) / price.Sys_peak
    Sys_maxDD = round(price.Sys_dd.min() * 100, 2)
    Sys_TotalReturn = round(((price.Sys_bal[-1] - price.Sys_bal[0]) / price.Sys_bal[-1]) * 100, 2)
    Sys_CAGR = round(((price.Sys_bal[-1] / price.Sys_bal[0]) ** (1 / years) - 1) * 100, 2)

    df = {'index title': ['Benchmark', 'System'], 'Total Return %': [Bench_TotalReturn, Sys_TotalReturn],
          'CAGR %': [Bench_CAGR, Sys_CAGR],
          'Maximum Drawdown %': [Bench_maxDD, Sys_maxDD]}
    backtest = pd.DataFrame(df).set_index('index title')
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f'Backtesting Period: {start} to {end}')
    plt.subplot(211)
    plt.plot(price.Bench_bal, label='Benchmark')
    plt.plot(price.Sys_bal, label='System')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(212)
    plt.plot(MACD, label='MACD')
    plt.plot(Signal, label='Signal')
    plt.axhline(y=0, color='b', linestyle='--')
    plt.title(f'a={a}, b={b}, c={c}')
    plt.legend()

    return backtest, price, fig


# %%
start = datetime.date(2015, 1, 1)
end = datetime.date(2021, 12, 2)
result, price, fig = macd('AAPL', start, end, a=14, b=30, c=15, positiveMACD=0)

plt.show()
