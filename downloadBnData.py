import requests
import pandas as pd
import numpy as np

# 下载U本位合约ETHUSDT日K线数据
url = "https://fapi.binance.com/fapi/v1/klines"
params = {
    "symbol": "ETHUSDT",
    "interval": "1d",
    # "startTime": ...,  # 可以设置起始时间
    # "endTime": ...,    # 可以设置结束时间
}
proxies = {
    "http": "http://127.0.0.1:7897",
    "https": "http://127.0.0.1:7897",
}

response = requests.get(url, params=params, proxies=proxies)
data = response.json()
print(data)
df = pd.DataFrame(data, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])

# 转换数据类型
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

# 收盘价均线
df['MA5']   = df['close'].rolling(window=5).mean()
df['MA10']  = df['close'].rolling(window=10).mean()
df['MA20']  = df['close'].rolling(window=20).mean()
df['MA50']  = df['close'].rolling(window=50).mean()
df['MA169'] = df['close'].rolling(window=169).mean()
df['MA200'] = df['close'].rolling(window=200).mean()

# 成交量均线
df['VMA5'] = df['volume'].rolling(window=5).mean()

# ATR(14) - 使用Wilder's EMA算法，与币安、TradingView一致
period = 14
tr_list = []
for i in range(len(df)):
    if i == 0:
        tr_list.append(df['high'][i] - df['low'][i])
    else:
        tr1 = df['high'][i] - df['low'][i]
        tr2 = abs(df['high'][i] - df['close'][i-1])
        tr3 = abs(df['low'][i] - df['close'][i-1])
        tr = max(tr1, tr2, tr3)
        tr_list.append(tr)
df['TR'] = tr_list
df['ATR14'] = np.nan
# 第一条ATR用前period个TR的均值
df.loc[period-1, 'ATR14'] = df['TR'][:period].mean()
# 后续用Wilder EMA
for i in range(period, len(df)):
    prev_atr = df.loc[i-1, 'ATR14']
    curr_tr = df.loc[i, 'TR']
    df.loc[i, 'ATR14'] = (prev_atr * (period - 1) + curr_tr) / period

# 布林带(20)
df['BOLL_MID'] = df['close'].rolling(window=20).mean()
df['BOLL_STD'] = df['close'].rolling(window=20).std()
df['BOLL_UPPER'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
df['BOLL_LOWER'] = df['BOLL_MID'] - 2 * df['BOLL_STD']

# RSI(14) - 用Wilder的EMA算法，和币安、TradingView一致
delta = df['close'].diff()
gain = pd.Series(np.where(delta > 0, delta, 0))
loss = pd.Series(np.where(delta < 0, -delta, 0))
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI14'] = 100 - (100 / (1 + rs))

# 保存为csv
# df.to_csv('ethusdt_futures_kline_ma_indicators.csv', index=False)

# 保存为json
df.to_json('ethusdt_futures_kline_ma_indicators.json', orient='records', force_ascii=False)