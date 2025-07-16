import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import json
from pathlib import Path

class TradingAdvisor:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio = {}
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.trade_history_file = self.data_dir / 'trade_history.json'
        self.load_trade_history()

    def load_trade_history(self):
        if self.trade_history_file.exists():
            with open(self.trade_history_file, 'r') as f:
                self.trade_history = json.load(f)
        else:
            self.trade_history = []

    def save_trade_history(self):
        with open(self.trade_history_file, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def fetch_data(self, symbol, period='3mo', interval='1d'):
        """Fetch historical data with enhanced indicators for swing trading"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data points for {symbol}. Got {len(df)}, need at least 50.")
        
        try:
            # Moving Averages
            df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
            df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
            
            # Momentum Indicators
            df['RSI'] = RSIIndicator(close=df['Close']).rsi()
            stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Trend Indicators
            if len(df) >= 30:  # ADX requires more data points
                adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
                df['ADX'] = adx.adx()
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Volatility Indicators
            bb = BollingerBands(close=df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mavg()
            df['BB_lower'] = bb.bollinger_lband()
            df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
            
            # Volume Indicators
            df['MFI'] = MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()
            df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            # VWAP
            vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
            df['VWAP'] = vwap.volume_weighted_average_price()
            
        except Exception as e:
            print(f"Warning: Error calculating some indicators for {symbol}: {str(e)}")
            # Ensure basic indicators are available even if advanced ones fail
            if 'SMA_20' not in df.columns:
                df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
            if 'RSI' not in df.columns:
                df['RSI'] = RSIIndicator(close=df['Close']).rsi()
            if 'BB_upper' not in df.columns:
                bb = BollingerBands(close=df['Close'])
                df['BB_upper'] = bb.bollinger_hband()
                df['BB_lower'] = bb.bollinger_lband()
            if 'VWAP' not in df.columns:
                vwap = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
                df['VWAP'] = vwap.volume_weighted_average_price()
        
        # Save data to CSV
        filename = self.data_dir / f'{symbol}_data.csv'
        df.to_csv(filename)
        return df

    def get_current_price(self, symbol):
        """Get the current price for a symbol"""
        df = self.fetch_data(symbol, period='1d', interval='1m')
        return df.iloc[-1]['Close']

    def analyze_symbol(self, symbol):
        """Enhanced analysis for swing trading"""
        df = self.fetch_data(symbol)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Trend Analysis
        trend_score = 0
        trend_score += 1 if latest['Close'] > latest['SMA_20'] else -1
        
        # Check if advanced indicators are available
        if 'SMA_50' in df.columns:
            trend_score += 1 if latest['Close'] > latest['SMA_50'] else -1
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            trend_score += 1 if latest['MACD'] > latest['MACD_Signal'] else -1
            
        trend_strength = 'Strong Up' if trend_score >= 2 else 'Weak Up' if trend_score > 0 else 'Strong Down' if trend_score <= -2 else 'Weak Down'
        
        # Volume Analysis
        volume_score = 0
        if 'MFI' in df.columns:
            volume_score += 1 if latest['MFI'] > 50 else -1
        if 'OBV' in df.columns:
            volume_score += 1 if latest['OBV'] > df['OBV'].mean() else -1
        volume_signal = 'Strong' if abs(volume_score) == 2 else 'Moderate' if abs(volume_score) == 1 else 'Weak'
        
        # Volatility Analysis
        volatility_status = 'HIGH' if latest['ATR'] > df['ATR'].mean() * 1.2 else 'LOW' if latest['ATR'] < df['ATR'].mean() * 0.8 else 'MODERATE'
        
        # VWAP Analysis
        vwap_signal = 'above' if latest['Close'] > latest['VWAP'] else 'below' if latest['Close'] < latest['VWAP'] else 'at'
        
        # Combined Signals
        signals = {
            'trend': trend_score,
            'momentum': 'bullish' if latest['RSI'] > 50 and latest['Stoch_K'] > latest['Stoch_D'] else 'bearish',
            'volume': volume_score,
            'macd': 'bullish' if latest['MACD'] > latest['MACD_Signal'] else 'bearish',
            'bb_position': 'oversold' if latest['Close'] < latest['BB_lower'] else 'overbought' if latest['Close'] > latest['BB_upper'] else 'neutral'
        }
        
        # Calculate probability score (enhanced)
        score = 0
        score += trend_score
        score += 1 if signals['momentum'] == 'bullish' else -1
        score += volume_score
        score += 1 if signals['macd'] == 'bullish' else -1
        score += 1 if signals['bb_position'] == 'oversold' else -1 if signals['bb_position'] == 'overbought' else 0
        vwap_score = 1 if latest['Close'] > latest['VWAP'] else -1
        score += vwap_score  # Integrate VWAP into the total score
        
        # Normalize probability (0 to 1)
        max_score = 7  # Maximum possible score
        probability = (score + max_score) / (2 * max_score)
        
        # Risk level assessment
        risk_level = 'HIGH' if volatility_status == 'HIGH' or abs(latest['RSI'] - 50) > 30 else 'MODERATE'
        
        # Generate swing trade specific advice
        swing_advice = self._generate_swing_advice(symbol, latest, signals, volatility_status)
        
        return {
            'symbol': symbol,
            'current_price': latest['Close'],
            'signals': signals,
            'buy_probability': probability,
            'recommendation': 'BUY' if probability > 0.6 else 'SELL' if probability < 0.4 else 'HOLD',
            'risk_level': risk_level,
            'trend_strength': trend_strength,
            'volume_signal': volume_signal,
            'volatility_status': volatility_status,
            'swing_advice': swing_advice,
            'trade_reasoning': self._generate_trade_reasoning(symbol, probability, signals, risk_level)
        }

    def _generate_swing_advice(self, symbol, latest, signals, volatility):
        """Generate specific advice for swing trading"""
        advice = []
        
        # Symbol specific considerations
        if symbol == 'BTC-USD':
            advice.append("Bitcoin shows high overnight volatility. Set tight stops 2-3% below entry.")
        elif symbol == 'TQQQ':
            advice.append("3x leveraged - use smaller position size and expect high volatility.")
        elif symbol == 'SQQQ':
            advice.append("Inverse 3x leveraged - best for bearish NASDAQ outlook only.")
        
        # Technical considerations
        if latest['RSI'] > 70:
            advice.append("Overbought conditions - consider taking profits or tight stops.")
        elif latest['RSI'] < 30:
            advice.append("Oversold conditions - watch for reversal signals.")
        
        if volatility == 'HIGH':
            advice.append("High volatility - reduce position size and widen stops.")
        
        return " ".join(advice)

    def _generate_trade_reasoning(self, symbol, probability, signals, risk_level):
        """Generate detailed reasoning for the trade decision"""
        if probability > 0.6:
            reason = f"Strong buy signal ({probability:.2%} probability) based on "
            positive_signals = []
            if signals['trend'] > 0:
                positive_signals.append("positive trend")
            if signals['momentum'] == 'bullish':
                positive_signals.append("bullish momentum")
            if signals['volume'] > 0:
                positive_signals.append("strong volume")
            reason += ", ".join(positive_signals)
        elif probability < 0.4:
            reason = f"Sell signal ({probability:.2%} probability) due to "
            negative_signals = []
            if signals['trend'] < 0:
                negative_signals.append("negative trend")
            if signals['momentum'] == 'bearish':
                negative_signals.append("bearish momentum")
            if signals['volume'] < 0:
                negative_signals.append("weak volume")
            reason += ", ".join(negative_signals)
        else:
            reason = f"Hold signal ({probability:.2%} probability) - mixed indicators"
        
        return f"{reason}. Risk level: {risk_level}"

    def paper_trade(self, symbol, action, amount):
        """Execute a paper trade"""
        if action not in ['buy', 'sell']:
            raise ValueError("Action must be 'buy' or 'sell'")
        
        df = self.fetch_data(symbol)
        current_price = df.iloc[-1]['Close']
        
        if action == 'buy':
            if amount > self.current_balance:
                raise ValueError("Insufficient funds")
            
            shares = amount / current_price
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + shares
            self.current_balance -= amount
            
        else:  # sell
            if symbol not in self.portfolio or self.portfolio[symbol] == 0:
                raise ValueError("No shares to sell")
            
            shares_to_sell = amount / current_price
            if shares_to_sell > self.portfolio[symbol]:
                raise ValueError("Not enough shares to sell")
            
            self.portfolio[symbol] -= shares_to_sell
            self.current_balance += amount
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'amount': amount,
            'price': current_price,
            'shares': shares if action == 'buy' else shares_to_sell,
            'balance_after': self.current_balance
        }
        
        self.trade_history.append(trade)
        self.save_trade_history()
        return trade

    def get_portfolio_value(self):
        """Calculate total portfolio value"""
        total_value = self.current_balance
        
        for symbol, shares in self.portfolio.items():
            if shares > 0:
                df = self.fetch_data(symbol)
                current_price = df.iloc[-1]['Close']
                total_value += shares * current_price
        
        return total_value

    def plot_analysis(self, symbol):
        """Create an interactive plot of the analysis"""
        df = self.fetch_data(symbol)
        
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add indicators
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower'))
        
        fig.update_layout(title=f'{symbol} Analysis', xaxis_title='Date', yaxis_title='Price')
        
        # Save the plot as HTML
        plot_file = self.data_dir / f'{symbol}_analysis.html'
        fig.write_html(str(plot_file))
        return plot_file
