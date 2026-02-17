import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
import xgboost as xgb
from datetime import datetime, timedelta
import sys
import os
import requests

# ==============================================================================
# 👇 ΡΥΘΜΙΣΕΙΣ (Βάλε εδώ τις μετοχές σου)
# ==============================================================================
SYMBOLS = [
    "INGA.AS",  # ING
    "NVDA",     # Nvidia
    "OPAP.AT"   # ΟΠΑΠ
]

START_DATE = "2023-01-01"
HISTORY_FILE = "history.csv"  # Το αρχείο που θα κρατάει τη μνήμη

# ==============================================================================

pd.options.mode.chained_assignment = None 

# --- 1. ΣΥΝΑΡΤΗΣΕΙΣ TELEGRAM & ΙΣΤΟΡΙΚΟΥ ---

def send_telegram_message(message):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("⚠️ Telegram credentials not found. Skipping message.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        requests.post(url, json=payload)
        print("✅ Telegram message sent!")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")

def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    # Δημιουργία κενού ιστορικού αν δεν υπάρχει
    return pd.DataFrame(columns=["Date", "Ticker", "Predicted_Return", "Direction", "Actual_Return", "Result"])

def update_history(df_history, ticker, date_str, pred_return, direction):
    # Α. Έλεγχος παλιών προβλέψεων (Αν κέρδισαν ή έχασαν)
    mask = (df_history['Ticker'] == ticker) & (df_history['Result'].isna())
    
    if mask.any():
        print(f"🔎 Checking past predictions for {ticker}...")
        try:
            # Κατεβάζουμε δεδομένα 5 ημερών για σιγουριά
            recent_data = yf.download(ticker, period="5d", progress=False, auto_adjust=False)
            
            # Ανάκτηση τιμών κλεισίματος (Close)
            if isinstance(recent_data.columns, pd.MultiIndex):
                close_prices = recent_data.xs('Close', level=0, axis=1).iloc[:, 0]
            else:
                close_prices = recent_data['Close']

            # Υπολογισμός πραγματικής απόδοσης (χθεσινό κλείσιμο vs προχθεσινό)
            if len(close_prices) >= 2:
                actual_return = close_prices.pct_change().iloc[-1]
                
                # Ενημέρωση των κενών εγγραφών
                indices = df_history[mask].index
                for idx in indices:
                    pred_dir = df_history.loc[idx, 'Direction']
                    
                    # Κανόνας Επιτυχίας: 
                    # Αν είπαμε ΠΑΝΩ και πήγε ΠΑΝΩ (>0) -> WIN
                    # Αν είπαμε ΚΑΤΩ και πήγε ΚΑΤΩ (<0) -> WIN
                    is_correct = (pred_dir == "UP" and actual_return > 0) or \
                                 (pred_dir == "DOWN" and actual_return < 0)
                    
                    df_history.loc[idx, 'Actual_Return'] = round(actual_return * 100, 2)
                    df_history.loc[idx, 'Result'] = "WIN" if is_correct else "LOSS"
                    print(f"   -> Previous prediction result: {'WIN' if is_correct else 'LOSS'}")
        except Exception as e:
            print(f"⚠️ Could not verify yesterday's prediction: {e}")

    # Β. Προσθήκη της σημερινής πρόβλεψης
    new_row = {
        "Date": date_str,
        "Ticker": ticker,
        "Predicted_Return": round(pred_return * 100, 2),
        "Direction": direction,
        "Actual_Return": None, # Θα συμπληρωθεί αύριο
        "Result": None         # Θα συμπληρωθεί αύριο
    }
    
    df_history = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)
    return df_history

def get_stats(df_history, ticker):
    # Υπολογισμός ποσοστού επιτυχίας
    ticker_history = df_history[df_history['Ticker'] == ticker].dropna(subset=['Result'])
    total = len(ticker_history)
    
    if total == 0: 
        return "New Bot 👶 (No stats yet)"
    
    wins = len(ticker_history[ticker_history['Result'] == 'WIN'])
    win_rate = (wins / total) * 100
    
    last_res = ticker_history.iloc[-1]['Result'] if total > 0 else "N/A"
    return f"{win_rate:.1f}% (Last: {last_res})"

# --- 2. Η ΚΥΡΙΑ ΚΛΑΣΗ ΑΝΑΛΥΣΗΣ (FULL VERSION) ---

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
        # ΑΥΤΗ ΕΙΝΑΙ Η ΠΛΗΡΗΣ ΕΚΔΟΣΗ ΜΕ ΟΛΟΥΣ ΤΟΥΣ ΔΕΙΚΤΕΣ
        df = self.data.copy()
        if len(df) < 60: return
        
        # 1. Βασικές Αποδόσεις
        df['Return'] = df['Close'].pct_change()
        
        # 2. Κινητοί Μέσοι (Trend)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_Ratio'] = df['Close'] / df['SMA_50']
        
        # 3. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Trend Momentum)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 5. ATR (Volatility)
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High']-df['Low'], (df['High']-prev_close).abs(), (df['Low']-prev_close).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']

        # 6. Advanced Statistics (Risk & Correlation)
        window = 60
        rolling_cov = df['Return'].rolling(window).cov(df['Market_Return'])
        rolling_var = df['Market_Return'].rolling(window).var()
        df['Beta'] = rolling_cov / rolling_var  # Πόσο ακολουθεί την αγορά
        df['Skewness'] = df['Return'].rolling(window).skew() # Κίνδυνος κατάρρευσης
        df['Correlation'] = df['Return'].rolling(window).corr(df['Market_Return'])

        # 7. Lag Features (Τι έγινε χθες, προχθές)
        df['Return_Lag1'] = df['Return'].shift(1)
        df['Return_Lag2'] = df['Return'].shift(2)
        df['Return_Lag3'] = df['Return'].shift(3)

        self.data = df.dropna()

    def optimize_model(self, X_train, y_train):
        # ΑΥΤΗ ΕΙΝΑΙ Η ΠΛΗΡΗΣ ΕΚΔΟΣΗ ΕΚΠΑΙΔΕΥΣΗΣ (GRID SEARCH)
        param_grid = {
            'n_estimators': [100, 200, 300], 
            'learning_rate': [0.01, 0.03, 0.05],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8]
        }
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)
        
        # Χρονολογικός διαχωρισμός (για να μην κλέβει από το μέλλον)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Αναζήτηση των καλύτερων παραμέτρων
        search = RandomizedSearchCV(xgb_model, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=tscv, verbose=0, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        
        return search.best_estimator_

    def run_prediction(self, history_df):
        ml_data = self.data.copy()
        
        # Στόχος: Η απόδοση της ΕΠΟΜΕΝΗΣ μέρας
        ml_data['Target'] = ml_data['Return'].shift(-1)
        ml_data = ml_data.dropna(subset=['Target'])
        
        # Όλα τα δεδομένα που θα δει το μοντέλο
        features = [
            'Return', 'SMA_Ratio', 'RSI', 'MACD', 'MACD_Signal', 
            'ATR_Ratio', 'Market_Return', 'Beta', 'Skewness', 'Correlation', 
            'Return_Lag1', 'Return_Lag2', 'Return_Lag3'
        ]
        
        X = ml_data[features]
        y = ml_data['Target']
        
        if len(X) == 0: return history_df

        # Διαχωρισμός σε Train/Test (κρατάμε το τελευταίο 20% για έλεγχο)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Εκπαίδευση Μοντέλου
        model = self.optimize_model(X_train, y_train)
        
        # Πρόβλεψη για ΑΥΡΙΟ
        last_row = self.data.iloc[[-1]][features] 
        current_price = self.data.iloc[-1]['Close']
        
        # Λήψη σημαντικών τιμών για την αναφορά
        last_beta = last_row['Beta'].values[0]
        last_skew = last_row['Skewness'].values[0]
        last_rsi = last_row['RSI'].values[0]
        
        pred_return = model.predict(last_row)[0]
        pred_price = current_price * (1 + pred_return)
        
        # Καθορισμός κατεύθυνσης
        direction = "UP" if pred_return > 0 else "DOWN"
        trend_emoji = "🟢" if pred_return > 0 else "🔴"
        trend_text = "BULLISH" if pred_return > 0 else "BEARISH"

        # Ενημέρωση Ιστορικού
        today_str = datetime.now().strftime('%Y-%m-%d')
        history_df = update_history(history_df, self.ticker, today_str, pred_return, direction)
        
        # Λήψη Στατιστικών Επιτυχίας
        stats_text = get_stats(history_df, self.ticker)

        # Δημιουργία Μηνύματος
        msg = (
            f"🤖 *DAILY UPDATE: {self.ticker}*\n"
            f"📅 Date: {last_row.index[0].strftime('%d-%m-%Y')}\n"
            f"🏆 *Win Rate: {stats_text}*\n"
            f"-------------------\n"
            f"📊 *Market Stats:*\n"
            f"• Price: {current_price:.2f}\n"
            f"• RSI: {last_rsi:.2f}\n"
            f"• Risk (Skew): {last_skew:.2f}\n"
            f"-------------------\n"
            f"🔮 *FORECAST:*\n"
            f"• Trend: {trend_emoji} {trend_text}\n"
            f"• Target: {pred_price:.2f} ({pred_return*100:+.2f}%)"
        )
        
        print(msg) 
        send_telegram_message(msg)
        return history_df

if __name__ == "__main__":
    print("🚀 Starting StockBot Analysis...")
    
    # Φόρτωση Ιστορικού (Μνήμη)
    history = load_history()
    
    today = datetime.now()
    tomorrow_date = today + timedelta(days=1)
    end_str = tomorrow_date.strftime('%Y-%m-%d')
    
    for ticker in SYMBOLS:
        try:
            print(f"\n⏳ Analyzing {ticker}...")
            bot = StockAnalyzer(ticker, START_DATE, end_str)
            if bot.get_data() is not None:
                bot.add_indicators()
                # Τρέχουμε την πρόβλεψη και ενημερώνουμε το ιστορικό
                history = bot.run_prediction(history)
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            continue

    # Αποθήκευση του ενημερωμένου ιστορικού στο αρχείο
    history.to_csv(HISTORY_FILE, index=False)
    print("💾 History saved successfully.")
