#!/usr/bin/env python
# coding: utf-8

# # LGBM Stock Selection Strategy
# 
# This notebook contains the following sections:
# - 1. Data Preparation: Calculating Daily and Monthly Returns
# - 2. Factor Construction: Market Cap Factor, Momentum Factor, Custom Factor
# - 3. LGBM Stock Selection Strategy
# - 4. Strategy Backtesting
# - 5. Strategy Analysis

# # Import libraries
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


# ## 1. Data Preparation: Calculating Daily and Monthly Returns

# Read data
df = pd.read_excel(r"C:\\Users\\13770\\Desktop\\data\\stock_data.xlsx")
df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
df = df.sort_values(by=['TICKER_SYMBOL', 'TRADE_DATE'])

# Daily returns
df['daily_return'] = df.groupby('TICKER_SYMBOL')['CLOSE'].pct_change()

# Monthly returns
df.set_index('TRADE_DATE', inplace=True)
monthly_close = df.groupby(['TICKER_SYMBOL', pd.Grouper(freq='ME')])['CLOSE'].last()
monthly_return = monthly_close.groupby('TICKER_SYMBOL').pct_change().reset_index()
monthly_return.rename(columns={'CLOSE': 'monthly_return'}, inplace=True)


# Export daily and monthly returns to Excel file
df_reset = df.reset_index()
df_reset[['TRADE_DATE', 'TICKER_SYMBOL', 'daily_return']].to_excel('daily_return.xlsx', index=False)
monthly_return.to_excel('monthly_return.xlsx', index=False)


# ## Factor Construction

# Market cap factor: ln(month-end market cap)
monthly_mv = df.groupby(['TICKER_SYMBOL', pd.Grouper(freq='ME')])['MV'].last().reset_index()
monthly_mv['ln_mv'] = np.log(monthly_mv['MV'])

# Momentum factor: past three months cumulative return
monthly_close = monthly_close.reset_index()
monthly_close['momentum'] = monthly_close.groupby('TICKER_SYMBOL')['CLOSE'].pct_change(periods=3)

# Volatility factor: past 20 days return standard deviation
df['volatility_20d'] = df.groupby('TICKER_SYMBOL')['daily_return'].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
monthly_vol = df.groupby(['TICKER_SYMBOL', pd.Grouper(freq='ME')])['volatility_20d'].last().reset_index()
monthly_vol.rename(columns={'volatility_20d': 'volatility'}, inplace=True)


# Merge factors
factors = pd.merge(monthly_mv, monthly_close, on=['TICKER_SYMBOL', 'TRADE_DATE'], how='outer')
factors = pd.merge(factors, monthly_vol, on=['TICKER_SYMBOL', 'TRADE_DATE'], how='outer')

# Export to Excel file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_path = f"factor_data_{timestamp}.xlsx"
factors.to_excel(excel_path, index=False)
print(f"Factor data exported to Excel file: {excel_path}")


# ## LGBM Stock Selection Strategy

# Build feature table
factors = monthly_mv.merge(monthly_close[['TICKER_SYMBOL', 'TRADE_DATE', 'momentum']], on = ['TICKER_SYMBOL', 'TRADE_DATE'], how = 'inner')
factors = factors.merge(monthly_return, on = ['TICKER_SYMBOL', 'TRADE_DATE'], how = 'inner')
factors = factors.dropna(subset=['ln_mv', 'momentum', 'monthly_return']).copy()

# Training and prediction
split_date = pd.to_datetime('2023-01-01')
train = factors[factors['TRADE_DATE'] < split_date].copy()
test = factors[factors['TRADE_DATE'] >= split_date].copy()

model = LGBMRegressor(force_col_wise = True, random_state = 42)

features = ['ln_mv', 'momentum']
target = 'monthly_return'
model.fit(train[features], train[target])
test.loc [:, 'predicted_return'] = model.predict(test[features])

# Select top stocks
def select_top(df, n=20, column='predicted_return'):
    return df.nlargest(n, column)

portfolio = (test.groupby('TRADE_DATE', group_keys = False).apply(lambda x: x.nlargest(20, 'predicted_return')).reset_index(drop=True))

print(f"Training set sample size: {len(train)}")
print(f"Test set sample size: {len(test)}")
print(f"Portfolio stock size: {len(portfolio)}")


# Export results to Excel
portfolio.to_excel("portfolio_results.xlsx", index=False)


# ## 4. Strategy Backtesting

# Merge returns
df = df_reset.reset_index(drop = True)
daily_returns = df[['TRADE_DATE', 'TICKER_SYMBOL', 'daily_return']].copy() 
portfolio_daily = portfolio.merge(daily_returns, on=['TRADE_DATE', 'TICKER_SYMBOL'], how='left')
portfolio_daily_grouped = (portfolio_daily.groupby('TRADE_DATE')['daily_return'].mean().reset_index(name='portfolio_return'))

market_mean = daily_returns['daily_return'].mean() # Fill missing values with mean
portfolio_daily_grouped['portfolio_return'] = portfolio_daily_grouped['portfolio_return'].fillna(market_mean)

print("\nMerged data preview:")
print(portfolio_daily_grouped.head())


# Index data comparison
portfolio_daily_grouped['TRADE_DATE'] = pd.to_datetime(portfolio_daily_grouped['TRADE_DATE'])

hs300 = pd.read_excel(r"C:\Users\13770\Desktop\data\000300_index_data.xlsx")
hs300['TRADE_DATE'] = pd.to_datetime(hs300['TRADE_DATE'])
hs300['hs300_return'] = hs300['Close'].pct_change()

full_months = pd.DataFrame({'TRADE_DATE': pd.date_range(start='2023-01-31', end='2023-12-31', freq='ME')})
returns_df = (full_months.merge(portfolio_daily_grouped, on='TRADE_DATE', how='left').merge(hs300[['TRADE_DATE', 'hs300_return']], on='TRADE_DATE', how='left'))

missing_months = returns_df[returns_df['portfolio_return'].isna()]['TRADE_DATE']
print(f"Missing months: {missing_months.dt.strftime('%Y-%m').tolist()}")
returns_df['portfolio_return'] = returns_df['portfolio_return'].ffill().bfill().fillna(0)
returns_df['hs300_return'] = returns_df['hs300_return'].ffill().bfill().fillna(0)

returns_df['portfolio_cum'] = (1 + returns_df['portfolio_return']).cumprod()
returns_df['hs300_cum'] = (1 + returns_df['hs300_return']).cumprod()

print("\nAfter fixing:")
apr_sep_data = returns_df.copy()
print(apr_sep_data[['TRADE_DATE', 'portfolio_return', 'hs300_return', 'portfolio_cum', 'hs300_cum']].to_string(index=False))


# Calculate four metrics
def calculate_metrics(returns_series, risk_free_rate=0.03, days_per_year=252):
    # Annualized return
    annualized_return = (1 + returns_series.mean())**days_per_year - 1

    # Annualized volatility
    annualized_vol = returns_series.std() * np.sqrt(days_per_year)

    # Sharpe ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

    # Maximum drawdown
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    max_drawdown = (peak - cum_returns).max()

    return {
        'Annualized return': annualized_return,
        'Annualized volatility': annualized_vol,
        'Sharpe ratio': sharpe_ratio,
        'Maximum drawdown': max_drawdown
    }


# Strategy portfolio metrics
portfolio_metrics = calculate_metrics(returns_df['portfolio_return'])

# HS300 metrics
hs300_metrics = calculate_metrics(returns_df['hs300_return'])

# Convert metrics to DataFrame
metrics_df = pd.DataFrame({
    'Metrics': ['Annualized return', 'Annualized volatility', 'Sharpe ratio', 'Maximum drawdown'],
    'Strategy portfolio': [
        f"{portfolio_metrics['Annualized return']:.2%}",
        f"{portfolio_metrics['Annualized volatility']:.2%}",
        f"{portfolio_metrics['Sharpe ratio']:.2f}",
        f"{portfolio_metrics['Maximum drawdown']:.2%}"
    ],
    'HS300': [
        f"{hs300_metrics['Annualized return']:.2%}",
        f"{hs300_metrics['Annualized volatility']:.2%}",
        f"{hs300_metrics['Sharpe ratio']:.2f}",
        f"{hs300_metrics['Maximum drawdown']:.2%}"
    ]
})

print("\n========== Metrics ==========")
print(metrics_df)

# Export results to Excel
with pd.ExcelWriter('strategy_backtesting_results.xlsx') as writer:
    returns_df.to_excel(writer, sheet_name='returns_data', index=False)
    metrics_df.to_excel(writer, sheet_name='metrics', index=False)

print("\nResults saved to: strategy_backtesting_results.xlsx")


try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False 
except:
    try:
        from matplotlib.font_manager import FontProperties
        font = FontProperties(fname='simhei.ttf') 
        rcParams['font.family'] = font.get_name()
    except:
        print("Warning: cannot load Chinese font, using default font")


# Plot
plt.figure(figsize = (16,8))
plt.plot(returns_df['TRADE_DATE'], returns_df['portfolio_cum'], label = 'Strategy portfolio', linewidth = 2)
plt.plot(returns_df['TRADE_DATE'], returns_df['hs300_cum'], label='HS300', linewidth = 2, linestyle = '--')

plt.title('Strategy cumulative return vs HS300', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative return', fontsize=12)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=-1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 

plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## 5. Strategy Analysis

# ### 5.1 What market conditions does the strategy perform well in? Why?
# The strategy performs well in market conditions with strong trends or early recovery periods, especially:
# 
# 1. Market overall up or recovery period (e.g., policy benefits, macroeconomic recovery)
# 
# 2. Market with significant "The Stronger the Stronger" effect between stocks (e.g., "winner takes all" effect)

# Specific scenarios:
# - Volatile upward or slow bull market
# - Market with obvious rotation structure (e.g. growth style, value style alternation)
# - Sector/factor driven market
# 
# Reason:
# 
# - LGBM is a non-linear model, skilled at identifying and utilizing complex factors
# 1. Unlike traditional linear models, LGBM can automatically capture the interaction and non-linear effects between multiple factors
# 2. When there are non-linear patterns such as "style effect" or "momentum transmission" in the market, LGBM model can better identify and utilize them
# 
# - Model is sensitive to "inertia" features, momentum factors are more effective in bull markets
# 1. In bull markets/slow bull markets, the momentum of strong stocks tends to continue
# 2. LGBM will capture the impact of momentum factors on future returns, so the selected stocks are more likely to continue to outperform
# 
# - LGBM can utilize information in "segmented markets"
# 1. When the market style switches from large-cap to mid-cap, or from consumer to technology, LGBM can identify the stronger style features through historical learning
# 
# - LGBM model can make better predictions when the "signal is valid", but the model's stability decreases when the market is volatile or experiencing a systemic downturn (high noise, momentum factors are less effective)

# 
# ### 5.2 How to improve the strategy?
# - Introduce more style factors, such as value, volatility, quality factors, etc.

# PE valuation reversal factor (approximated by 1/PE)
df['inv_pe'] = 1 / df['CLOSE']  # If there is a PE field, it can be used directly

# ROE/ROA quality factors (requires fundamental data)

# - Add more algorithm fusion at the model level, such as XGBoost, neural network

# Model training
lgbm = LGBMRegressor()
xgb = XGBRegressor()
nn = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=500)

lgbm.fit(train[features], train[target])
xgb.fit(train[features], train[target])
nn.fit(train[features], train[target])

# Simple average fusion
test['pred_xgb'] = xgb.predict(test[features])
test['pred_nn'] = nn.predict(test[features])

test['predicted_return'] = test[['pred_lgbm', 'pred_xgb', 'pred_nn']].mean(axis=1)


# - Add turnover control, industry neutralization, and risk control module


# # Control turnover
# Record last month's holdings
holdings = {}
turnover_record = []

for date, group in test.groupby('TRADE_DATE'):
    top_stocks = group.nlargest(20, 'predicted_return')['TICKER_SYMBOL'].tolist()
    prev = holdings.get('prev', [])
    turnover = len(set(top_stocks) - set(prev)) / 20
    turnover_record.append({'TRADE_DATE': date, 'turnover': turnover})
    holdings['prev'] = top_stocks


# # Industry neutralization
# If there is an industry column, it can be implemented by groupby + demean
factors['momentum_ind_neutral'] = factors.groupby(['TRADE_DATE', 'industry'])['momentum'].transform(lambda x: x - x.mean())


# - Simulate real transaction costs

# Cost: set double-sided transaction fee to 0.1%
cost_rate = 0.001

# Introduce cost in daily returns
portfolio_daily['prev_ticker'] = portfolio_daily.groupby('TRADE_DATE')['TICKER_SYMBOL'].shift(1)
portfolio_daily['turnover_flag'] = (portfolio_daily['TICKER_SYMBOL'] != portfolio_daily['prev_ticker']).astype(int)

# Assume turnover occurs on rebalancing days, charge cost
portfolio_daily['cost'] = portfolio_daily['turnover_flag'] * cost_rate
portfolio_daily['net_return'] = portfolio_daily['daily_return'] - portfolio_daily['cost']