# Αρχείο: main.py
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
import xgboost as xgb
from datetime import datetime, timedelta
import sys

# --- ΡΥΘΜΙΣΕΙΣ ---
SYMBOL = "INGA.AS"  
START_DATE = "2023-01-01"

pd.options.mode.chained_assignment = None 

class StockAnalyzer:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None

    def determine_market_index(self):
        ticker = self.ticker.upper()
        market_map = {
            '.AS': '^AEX', '.DE': '^GDAXI', '.PA': '^FCHI', 
            '.L': '^FTSE', '.MI': 'FTSEMIB.MI', '.MC': '^IBEX', '.AT': '^ATG'
        }
        for suffix, index_ticker in market_map.items():
            if ticker.endswith(suffix): return index_ticker
        
        european_suffixes = ['.BR', '.LS', '.VI', '.ST', '.HE', '.CO']
        for suffix in european_suffixes:
            if ticker.endswith(suffix): return '^STOXX50E'
        return '^GSPC'

    def _flatten_yfinance(self, raw):
        df = pd.DataFrame()
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df['Close'] = raw.xs('Close', level=0, axis=1).iloc[:, 0]
                df['High'] = raw.xs('High', level=0, axis=1).iloc[:, 0]
                df['Low'] = raw.xs('Low', level=0, axis=1).iloc[:, 0]
            else:
                df['Close'] = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
                df['High'] = raw['High'] if 'High' in raw.columns else raw['Close']
                df['Low'] = raw['Low'] if 'Low' in raw.columns else raw['Close']
            for col in df.columns: df[col] = df[col].astype(float)
            return df
        except: return None

    def get_data(self):
        market_ticker = self.determine_market_index()
        print(f"--- DATA FETCH: {self.ticker} vs {market_ticker} ---")
        try:
            raw = yf.download(self.ticker, start=self.start, end=self.end, progress=False, auto_adjust=False)
            df = self._flatten_yfinance(raw)
            if df is None or df.empty: return None
            
            market_raw = yf.download(market_ticker, start=self.start, end=self.end, progress=False, auto_adjust=False)
            df_market = self._flatten_yfinance(market_raw)
            
            if df_market is not None and not df_market.empty:
                df_market['Market_Return'] = df_market['Close'].pct_change()
                df = df.join(df_market[['Market_Return']], how='left')
                df['Market_Return'] = df['Market_Return'].ffill().fillna(0)
            else:
                df['Market_Return'] = 0.0

            self.data = df
            return self.data
        except Exception as e:
            print(f"Error: {e}")
            return None

    def add_indicators(self):
        df = self.data.copy()
        if len(df) < 60: return
        df['Return'] = df['Close'].pct_change()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_Ratio'] = df['Close'] / df['SMA_50']
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High']-df['Low'], (df['High']-prev_close).abs(), (df['Low']-prev_close).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']

        window = 60
        rolling_cov = df['Return'].rolling(window).cov(df['Market_Return'])
        rolling_var = df['Market_Return'].rolling(window).var()
        df['Beta'] = rolling_cov / rolling_var
        df['Skewness'] = df['Return'].rolling(window).skew()
        df['Correlation'] = df['Return'].rolling(window).corr(df['Market_Return'])

        df['Return_Lag1'] = df['Return'].shift(1)
        df['Return_Lag2'] = df['Return'].shift(2)
        df['Return_Lag3'] = df['Return'].shift(3)

        self.data = df.dropna()

    def optimize_model(self, X_train, y_train):
        param_grid = {
            'n_estimators': [100, 200], # Λίγο πιο ελαφρύ για το server
            'learning_rate': [0.03, 0.05],
            'max_depth': [3, 4],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(xgb_model, param_grid, n_iter=5, scoring='neg_mean_squared_error', cv=tscv, verbose=0, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        return search.best_estimator_

    def run_prediction(self):
        ml_data = self.data.copy()
        ml_data['Target'] = ml_data['Return'].shift(-1)
        ml_data = ml_data.dropna(subset=['Target'])
        
        features = [
            'Return', 'SMA_Ratio', 'RSI', 'MACD', 'MACD_Signal', 
            'ATR_Ratio', 'Market_Return', 'Beta', 'Skewness', 'Correlation', 
            'Return_Lag1', 'Return_Lag2', 'Return_Lag3'
        ]
        
        X = ml_data[features]
        y = ml_data['Target']
        
        if len(X) == 0: return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = self.optimize_model(X_train, y_train)
        
        last_row = self.data.iloc[[-1]][features] 
        current_price = self.data.iloc[-1]['Close']
        
        last_beta = last_row['Beta'].values[0]
        last_skew = last_row['Skewness'].values[0]
        last_rsi = last_row['RSI'].values[0]
        
        pred_return = model.predict(last_row)[0]
        pred_price = current_price * (1 + pred_return)
        
        print("\n" + "="*40)
        print(f"🤖 AUTOMATED REPORT FOR: {self.ticker}")
        print(f"📅 Date: {last_row.index[0].strftime('%d-%m-%Y')}")
        print("="*40)
        print(f"📊 Market Stats:")
        print(f"   • Current Price: {current_price:.2f}")
        print(f"   • RSI:           {last_rsi:.2f}")
        print(f"   • Beta (Risk):   {last_beta:.2f}")
        print(f"   • Skewness:      {last_skew:.2f}")
        print("-" * 40)
        print(f"🔮 PREDICTION FOR TOMORROW:")
        print(f"   • Change: {pred_return*100:+.2f}%")
        print(f"   • Target: {pred_price:.2f}")
        print(f"   • Trend:  {'🟢 BULLISH (ΑΝΟΔΟΣ)' if pred_return > 0 else '🔴 BEARISH (ΠΤΩΣΗ)'}")
        print("="*40 + "\n")

if __name__ == "__main__":
    today = datetime.now()
    tomorrow_date = today + timedelta(days=1)
    end_str = tomorrow_date.strftime('%Y-%m-%d')
    
    bot = StockAnalyzer(SYMBOL, START_DATE, end_str)
    if bot.get_data() is not None:
        bot.add_indicators()
        bot.run_prediction()
