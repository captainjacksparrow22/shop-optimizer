import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import json
import logging
import time
from pathlib import Path
import time
import logging

class DayTradingAdvisor:
    def __init__(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio = {}
        self.base_path = Path(r'C:\Users\Nathan\Documents\Python\shop-optimizer\stock-analyzer-2')
        self.data_dir = self.base_path / 'data'
        self.data_dir.mkdir(exist_ok=True)
        self.trade_history_file = self.data_dir / 'trade_history.json'
        self.performance_file = self.data_dir / 'performance_history.json'
        self.load_trade_history()
        self.load_performance_history()
        
        # Track daily and total P/L
        self.daily_pl = 0
        self.total_pl = 0
        self.winning_trades = 0
        self.total_trades = 0

    def load_trade_history(self):
        if self.trade_history_file.exists():
            with open(self.trade_history_file, 'r') as f:
                self.trade_history = json.load(f)
        else:
            self.trade_history = []

    def save_trade_history(self):
        with open(self.trade_history_file, 'w') as f:
            json.dump(self.trade_history, f, indent=4)

    def load_performance_history(self):
        """Load performance history from file"""
        if self.performance_file.exists():
            with open(self.performance_file, 'r') as f:
                data = json.load(f)
                self.total_pl = data.get('total_pl', 0)
                self.winning_trades = data.get('winning_trades', 0)
                self.total_trades = data.get('total_trades', 0)
        else:
            self.total_pl = 0
            self.winning_trades = 0
            self.total_trades = 0

    def save_performance_history(self):
        """Save performance history to file"""
        data = {
            'total_pl': self.total_pl,
            'winning_trades': self.winning_trades,
            'total_trades': self.total_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.performance_file, 'w') as f:
            json.dump(data, f, indent=4)

    def update_performance(self, trade):
        """Update performance metrics after a trade"""
        if trade['action'] == 'SELL':
            # Calculate P/L for this trade
            entry_price = None
            for hist_trade in reversed(self.trade_history[:-1]):  # Exclude current trade
                if hist_trade['symbol'] == trade['symbol'] and hist_trade['action'] == 'BUY':
                    entry_price = hist_trade['price']
                    break
            
            if entry_price is not None:
                pl = (trade['price'] - entry_price) * trade['shares']
                self.daily_pl += pl
                self.total_pl += pl
                self.total_trades += 1
                if pl > 0:
                    self.winning_trades += 1
                
                self.save_performance_history()
                return pl
        return 0

    def fetch_data(self, symbol, lookback_hours=4):
        """Fetch multiple timeframe data for triple screen analysis"""
        ticker = yf.Ticker(symbol)
        end_time = datetime.now()
        
        try:
            # Fetch data for each timeframe with appropriate lookback periods
            # For 1h data, we need more history to get enough bars
            start_1h = end_time - timedelta(days=2)  # 48 hours for hourly data
            df_1h = ticker.history(start=start_1h, end=end_time, interval='1h')
            time.sleep(1)  # Rate limiting pause
            
            # For 15m data, look back 12 hours
            start_15m = end_time - timedelta(hours=12)
            df_15m = ticker.history(start=start_15m, end=end_time, interval='15m')
            time.sleep(1)  # Rate limiting pause
            
            # For 5m data, look back 4 hours
            start_5m = end_time - timedelta(hours=4)
            df_5m = ticker.history(start=start_5m, end=end_time, interval='5m')
            
            # Ensure we have enough data points
            min_points = 20  # Minimum required for indicators
            data_points = {
                '5m': len(df_5m),
                '15m': len(df_15m),
                '1h': len(df_1h)
            }
            
            if any(points < min_points for points in data_points.values()):
                raise ValueError(f"Insufficient data points. 5m: {data_points['5m']}, "
                               f"15m: {data_points['15m']}, 1h: {data_points['1h']}")
            
            logging.info(f"Successfully fetched data for {symbol}. Points - "
                        f"5m: {data_points['5m']}, 15m: {data_points['15m']}, "
                        f"1h: {data_points['1h']}")
            
            return df_5m, df_15m, df_1h
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None

    def _add_indicators(self, df, suffix=''):
        """Add technical indicators with optional suffix for timeframe identification"""
        try:
            # Use shorter periods for limited data scenarios
            ema_long_period = min(50, len(df) - 10) if len(df) > 20 else 20
            ema_short_period = min(20, len(df) - 5) if len(df) > 10 else 10
            
            # Trend Indicators
            df[f'EMA_50{suffix}'] = EMAIndicator(close=df['Close'], window=ema_long_period).ema_indicator()
            df[f'EMA_20{suffix}'] = EMAIndicator(close=df['Close'], window=ema_short_period).ema_indicator()
            
            # RSI for momentum
            rsi_period = min(14, len(df) - 5) if len(df) > 15 else 7
            df[f'RSI{suffix}'] = RSIIndicator(close=df['Close'], window=rsi_period).rsi()
            
            # Stochastic RSI
            df[f'Stoch_RSI{suffix}'] = StochRSIIndicator(close=df['Close'], window=rsi_period).stochrsi()
            
            # MACD for trend confirmation
            macd = MACD(close=df['Close'])
            df[f'MACD{suffix}'] = macd.macd()
            df[f'MACD_Signal{suffix}'] = macd.macd_signal()
            
            # Bollinger Bands for volatility
            bb_period = min(20, len(df) - 5) if len(df) > 20 else 10
            bb = BollingerBands(close=df['Close'], window=bb_period)
            df[f'BB_upper{suffix}'] = bb.bollinger_hband()
            df[f'BB_lower{suffix}'] = bb.bollinger_lband()
            
            # ATR for stop loss calculation
            atr_period = min(14, len(df) - 3) if len(df) > 14 else 7
            df[f'ATR{suffix}'] = AverageTrueRange(high=df['High'], low=df['Low'], 
                                                close=df['Close'], window=atr_period).average_true_range()
            
            # Volume analysis
            df[f'VWAP{suffix}'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], 
                                                            volume=df['Volume']).volume_weighted_average_price()
            
            # Trend signal (1 for bullish, -1 for bearish, 0 for neutral)
            df[f'trend_signal{suffix}'] = self._calculate_trend_signal(df, suffix)
            
            logging.info(f"Successfully calculated indicators for suffix '{suffix}' with {len(df)} data points")
            
        except Exception as e:
            logging.error(f"Error calculating indicators for suffix '{suffix}': {str(e)}")
            # Add minimal fallback indicators
            df[f'trend_signal{suffix}'] = pd.Series(0, index=df.index)
        
        return df
        
    def _calculate_trend_signal(self, df, suffix=''):
        """Calculate trend signal based on multiple indicators"""
        signal = pd.Series(0, index=df.index)
        
        try:
            # EMA trend
            if f'EMA_50{suffix}' in df.columns:
                signal += np.where(df['Close'] > df[f'EMA_50{suffix}'], 1, -1)
            
            # MACD
            if f'MACD{suffix}' in df.columns and f'MACD_Signal{suffix}' in df.columns:
                signal += np.where(df[f'MACD{suffix}'] > df[f'MACD_Signal{suffix}'], 1, -1)
            
            # RSI
            if f'RSI{suffix}' in df.columns:
                signal += np.where((df[f'RSI{suffix}'] > 50) & (df[f'RSI{suffix}'] < 70), 1,
                                  np.where((df[f'RSI{suffix}'] < 50) & (df[f'RSI{suffix}'] > 30), -1, 0))
            
            # StochRSI
            if f'Stoch_RSI{suffix}' in df.columns:
                signal += np.where(df[f'Stoch_RSI{suffix}'] > 0.8, -1, 
                          np.where(df[f'Stoch_RSI{suffix}'] < 0.2, 1, 0))

            # Normalize to -1, 0, 1
            signal = np.where(signal > 1, 1, np.where(signal < -1, -1, 0))
            
        except Exception as e:
            logging.error(f"Error calculating trend signal with suffix '{suffix}': {str(e)}")
            # Return neutral signal if calculation fails
            signal = pd.Series(0, index=df.index)
        
        return signal
        
    def _resample_signal(self, signal, target_index):
        """Resample higher timeframe signals to 5-minute timeframe"""
        try:
            # Forward fill the signal to match the target index
            resampled = signal.reindex(target_index, method='ffill')
            
            # If there are still NaN values, backfill them
            if resampled.isna().any():
                resampled = resampled.fillna(method='bfill')
            
            # If still NaN, fill with 0 (neutral)
            resampled = resampled.fillna(0)
            
            logging.info(f"Resampled signal from {len(signal)} to {len(resampled)} points")
            return resampled
            
        except Exception as e:
            logging.error(f"Error resampling signal: {str(e)}")
            # Return neutral signal as fallback
            return pd.Series(0, index=target_index)

    def calculate_stop_loss(self, entry_price, atr, is_long=True):
        """Calculate stop loss based on ATR"""
        atr_multiplier = 1.5
        if is_long:
            return entry_price - (atr * atr_multiplier)
        return entry_price + (atr * atr_multiplier)

    def analyze_symbol(self, symbol):
        """Triple screen analysis for high-probability trades"""
        df_5m, df_15m, df_1h = self.fetch_data(symbol)
        
        # Check if data fetching was successful
        if df_5m is None or df_15m is None or df_1h is None:
            logging.error(f"Unable to perform analysis for {symbol} due to missing data")
            return {
                'symbol': symbol,
                'recommendation': 'HOLD',
                'error': 'Data fetch failed'
            }
            
        # Process the data
        df_1h = self._add_indicators(df_1h, suffix='_1h')
        df_15m = self._add_indicators(df_15m, suffix='_15m')
        df_5m = self._add_indicators(df_5m, suffix='_5m')
        
        # Debug: Check if trend_signal columns exist
        logging.info(f"1H columns: {list(df_1h.columns)}")
        logging.info(f"15M columns: {list(df_15m.columns)}")
        
        # Check if trend_signal columns exist before trying to resample
        if 'trend_signal_1h' not in df_1h.columns:
            logging.error("trend_signal_1h column not found in 1H dataframe")
            df_1h['trend_signal_1h'] = pd.Series(0, index=df_1h.index)
            
        if 'trend_signal_15m' not in df_15m.columns:
            logging.error("trend_signal_15m column not found in 15M dataframe")
            df_15m['trend_signal_15m'] = pd.Series(0, index=df_15m.index)
        
        # Merge signals from higher timeframes into 5m dataframe
        df = df_5m.copy()
        df['trend_1h'] = self._resample_signal(df_1h['trend_signal_1h'], df.index)
        df['trend_15m'] = self._resample_signal(df_15m['trend_signal_15m'], df.index)
        
        latest = df.iloc[-1]
        logging.info(f"Analyzing data for {symbol} up to timestamp: {latest.name}")
        prev = df.iloc[-2]
        
        # 1. Higher Timeframe Trend (1-hour)
        trend_1h = latest['trend_1h']
        
        # 2. Medium Timeframe Confirmation (15-min)
        trend_15m = latest['trend_15m']
        
        # 3. Entry Timing (5-min)
        # Calculate current momentum and volatility
        momentum_5m = 1 if latest['MACD_5m'] > latest['MACD_Signal_5m'] else -1
        vol_condition = latest['Close'] > latest['VWAP_5m']
        
        # Combined analysis
        trend_score = trend_1h + trend_15m
        momentum_score = momentum_5m + (1 if vol_condition else -1)
        volume_score = 1 if latest['Volume'] > df['Volume'].rolling(10).mean().iloc[-1] else -1
        
        # Final probability calculation
        total_score = trend_score + momentum_score + volume_score
        max_score = 5  # Maximum possible score
        probability = (total_score + max_score) / (2 * max_score)
        
        # Stop loss calculation using ATR
        atr_multiplier = 2.0  # Wider for crypto's volatility
        stop_loss = latest['Close'] - (latest['ATR_5m'] * atr_multiplier) if probability > 0.5 else \
                   latest['Close'] + (latest['ATR_5m'] * atr_multiplier)
        
        # Determine recommendation
        if trend_1h > 0 and trend_15m > 0 and momentum_5m > 0:
            recommendation = 'BUY'
        elif trend_1h < 0 and trend_15m < 0 and momentum_5m < 0:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
            
        # Risk level based on volatility and trend alignment
        risk_level = 'HIGH' if abs(latest['ATR_5m']) > df['ATR_5m'].mean() * 1.5 else 'MODERATE'
        
        trend_strength = self._calculate_trend_strength(trend_1h, trend_15m)
        
        # Volume Analysis
        volume_score = 0
        volume_score += 1 if latest['Volume'] > df['Volume'].rolling(10).mean().iloc[-1] else -1
        volume_score += 1 if latest['Close'] > latest['VWAP_5m'] else -1
        
        # Momentum
        momentum_score = 0
        momentum_score += 1 if 30 < latest['RSI_5m'] < 70 else -1
        momentum_score += 1 if latest['Stoch_RSI_5m'] > 0.2 and latest['Stoch_RSI_5m'] < 0.8 else 0
        
        # Volatility Check
        volatility_status = 'HIGH' if latest['ATR_5m'] > df['ATR_5m'].mean() * 1.2 else 'LOW'
        
        # Combined Analysis
        total_score = trend_score + volume_score + momentum_score
        max_possible_score = 7
        probability = (total_score + max_possible_score) / (2 * max_possible_score)
        
        # Stop Loss
        suggested_stop = self.calculate_stop_loss(latest['Close'], latest['ATR_5m'], probability > 0.5)
        
        return {
            'symbol': symbol,
            'current_price': latest['Close'],
            'probability': probability,
            'recommendation': 'BUY' if probability > 0.65 else 'SELL' if probability < 0.35 else 'HOLD',
            'stop_loss': suggested_stop,
            'risk_level': 'HIGH' if volatility_status == 'HIGH' else 'MODERATE',
            'trend_strength': 'Strong Up' if trend_score >= 2 else 'Weak Up' if trend_score > 0 else 'Strong Down' if trend_score <= -2 else 'Weak Down',
            'volume_quality': 'Good' if volume_score > 0 else 'Poor',
            'trade_reasoning': self._generate_trade_reasoning(probability, trend_score, volume_score, momentum_score)
        }

    def _generate_trade_reasoning(self, trend_1h, trend_15m, momentum_5m, vol_condition):
        """Generate detailed reasoning for the trade decision based on triple screen analysis"""
        reasons = []
        
        # Higher timeframe analysis
        if trend_1h > 0:
            reasons.append("1H Timeframe: Bullish trend")
        elif trend_1h < 0:
            reasons.append("1H Timeframe: Bearish trend")
        else:
            reasons.append("1H Timeframe: Neutral trend")
            
        # Medium timeframe analysis
        if trend_15m > 0:
            reasons.append("15M Timeframe: Upward momentum")
        elif trend_15m < 0:
            reasons.append("15M Timeframe: Downward momentum")
        else:
            reasons.append("15M Timeframe: Consolidating")
            
        # Entry timing (5-min)
        if momentum_5m > 0:
            reasons.append("5M Timeframe: Positive momentum")
        else:
            reasons.append("5M Timeframe: Negative momentum")
            
        if vol_condition:
            reasons.append("Volume: Above VWAP, showing strength")
        else:
            reasons.append("Volume: Below VWAP, showing weakness")
        
        return ". ".join(reasons)

    def execute_trade(self, symbol, recommendation, price, stop_loss):
        """Execute a paper trade with stop loss"""
        trade = None
        
        if recommendation == 'BUY' and self.current_balance > 100:
            # Risk 2% per trade
            risk_amount = self.current_balance * 0.02
            stop_distance = abs(price - stop_loss)
            position_size = risk_amount / stop_distance
            investment = position_size * price
            
            if investment > self.current_balance:
                investment = self.current_balance * 0.95  # Use 95% max
            
            shares = investment / price
            self.portfolio[symbol] = self.portfolio.get(symbol, 0) + shares
            self.current_balance -= investment
            
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'investment': investment,
                'stop_loss': stop_loss,
                'balance_after': self.current_balance
            }
            
        elif recommendation == 'SELL' and symbol in self.portfolio:
            shares = self.portfolio[symbol]
            if shares > 0:
                sale_amount = shares * price
                self.portfolio[symbol] = 0
                self.current_balance += sale_amount
                
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares,
                    'amount': sale_amount,
                    'balance_after': self.current_balance
                }
        
        if trade is not None:
            self.trade_history.append(trade)
            self.save_trade_history()
            
        return trade

    def plot_analysis(self, symbol):
        """Create an interactive plot focused on short-term trading"""
        df_5m, df_15m, df_1h = self.fetch_data(symbol)
        
        if df_5m is None or df_15m is None or df_1h is None:
            logging.error(f"Unable to create plot for {symbol} due to missing data")
            return None
            
        # Process the 5-minute data for plotting
        df = self._add_indicators(df_5m.copy(), suffix='_5m')
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add indicators
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20_5m'], name='EMA 20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50_5m'], name='EMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP_5m'], name='VWAP'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper_5m'], name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower_5m'], name='BB Lower'))
        
        # Volume bars and StochRSI on a secondary axis
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_RSI_5m'], name='Stoch RSI', yaxis='y3'))
        
        # Layout
        fig.update_layout(
            title=f'{symbol} 5-Minute Analysis',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title='Stoch RSI',
                overlaying='y',
                side='right',
                position=0.85,
                anchor='free'
            ),
            xaxis_title='Time'
        )
        
        # Save the plot
        plot_file = self.data_dir / f'{symbol}_5min_analysis.html'
        fig.write_html(str(plot_file))
        return plot_file

    def _calculate_trend_strength(self, trend_1h, trend_15m):
        """Calculate trend strength based on multiple timeframes"""
        if trend_1h > 0 and trend_15m > 0:
            return 'Strong Up'
        elif trend_1h > 0 and trend_15m == 0:
            return 'Weak Up'
        elif trend_1h < 0 and trend_15m < 0:
            return 'Strong Down'
        elif trend_1h < 0 and trend_15m == 0:
            return 'Weak Down'
        else:
            return 'Neutral'
