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
import logging

class TradingAdvisor:
    def __init__(self, initial_balance=1000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.portfolio_value = initial_balance
        self.daily_pl = 0
        
        self.base_path = Path(r'C:\Users\Nathan\Documents\Python\shop-optimizer\stock-analyzer')
        self.data_dir = self.base_path / 'data'
        self.data_dir.mkdir(exist_ok=True)
        self.portfolio_file = self.data_dir / 'portfolio.json'
        self.trade_history_file = self.data_dir / 'trade_history.json'
        self.log_dir = self.base_path / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f'trading_log_{datetime.now().strftime("%Y-%m-%d")}.txt'
        self._configure_logging()
        self.load_portfolio()

    def _configure_logging(self):
        """Configure logging to file and console."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def load_portfolio(self):
        """Load portfolio from a JSON file."""
        if self.portfolio_file.exists():
            with open(self.portfolio_file, 'r') as f:
                self.positions = json.load(f)
        else:
            self.positions = {}

    def save_portfolio(self):
        """Save portfolio to a JSON file."""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.positions, f, indent=4)

    def load_trade_history(self):
        """Load trade history from a JSON file."""
        if self.trade_history_file.exists():
            with open(self.trade_history_file, 'r') as f:
                self.trade_history = json.load(f)
        else:
            self.trade_history = []

    def save_trade_history(self):
        """Save trade history to a JSON file."""
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

    def update_portfolio_value(self, symbol_prices):
        """Update the value of current holdings based on latest prices and calculate P/L."""
        self.portfolio_value = self.balance
        unrealized_pl = 0
        for symbol, position in self.positions.items():
            if symbol in symbol_prices:
                current_price = symbol_prices[symbol]
                current_value = position['shares'] * current_price
                investment_cost = position['investment']
                position_pl = current_value - investment_cost
                unrealized_pl += position_pl
                self.portfolio_value += current_value
        
        self.daily_pl = self.portfolio_value - self.initial_balance
        return unrealized_pl

    def execute_trade(self, symbol, recommendation, price, reasoning):
        """Execute a paper trade based on the recommendation."""
        action = 'buy' if recommendation == 'BUY' else 'sell'
        investment_amount = self.balance * 0.1  # Invest 10% of balance
        
        if action == 'buy':
            if investment_amount > self.balance:
                logging.warning(f"Not enough balance to invest ${investment_amount:.2f} in {symbol}. Skipping trade.")
                return

            shares = investment_amount / price
            self.positions[symbol] = self.positions.get(symbol, {'shares': 0, 'investment': 0})
            self.positions[symbol]['shares'] += shares
            self.positions[symbol]['investment'] += investment_amount
            self.balance -= investment_amount
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'buy',
                'amount': investment_amount,
                'price': price,
                'shares': shares,
                'balance_after': self.balance
            }
        else:  # sell
            if symbol not in self.positions or self.positions[symbol]['shares'] == 0:
                logging.info(f"No position in {symbol} to sell.")
                return
            
            shares_to_sell = self.positions[symbol]['shares']  # Sell all shares
            amount = shares_to_sell * price
            self.balance += amount
            self.positions.pop(symbol) # Remove position after selling
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'sell',
                'amount': amount,
                'price': price,
                'shares': shares_to_sell,
                'balance_after': self.balance
            }
        
        self.save_portfolio()
        self.log_trade(trade)

    def analyze(self, symbols):
        """Analyze the given symbols and update portfolio based on recommendations."""
        logging.info("\nAnalyzing market conditions...")

        symbol_prices = {symbol: yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1] for symbol in symbols}

        for symbol in symbols:
            try:
                logging.info(f"\n==================================================")
                logging.info(f"{symbol} Swing Trade Analysis:")
                data = self.fetch_data(symbol)
                analysis = self.analyze_symbol(symbol)
                recommendation = analysis['recommendation']
                current_price = analysis['current_price']
                probability = analysis['buy_probability']
                reasoning = analysis['trade_reasoning']
                
                # Log the analysis result
                logging.info(f"\nAnalysis for {symbol}:")
                logging.info(f"Current Price: ${current_price:.2f}")
                logging.info(f"Recommendation: {recommendation}")
                logging.info(f"Buy Probability: {probability:.2%}")
                logging.info(f"Reasoning: {reasoning}")
                
                # Execute trade if probability is high enough
                if probability > 0.7 and recommendation == 'BUY':
                    self.execute_trade(symbol, recommendation, current_price, reasoning)
                elif recommendation == 'SELL' and symbol in self.positions:
                     self.execute_trade(symbol, recommendation, current_price, reasoning)

            except Exception as e:
                logging.error(f"Error analyzing {symbol}: {e}")
        
        unrealized_pl = self.update_portfolio_value(symbol_prices)
        self.log_summary(unrealized_pl)

    def log_trade(self, trade):
        """Log a single trade."""
        logging.info(f"Trade executed: {trade['action'].capitalize()} {trade['shares']:.4f} shares of {trade['symbol']} at ${trade['price']:.2f} each. Amount: ${trade['amount']:.2f}.")
        logging.info(f"Portfolio updated: Balance ${self.balance:.2f}, Positions: {len(self.positions)}")

    def log_summary(self, unrealized_pl):
        """Log the end-of-day summary."""
        logging.info("\n==================================================")
        logging.info("=== End of Day Summary ===")
        logging.info(f"Current portfolio value: ${self.portfolio_value:.2f}")
        
        daily_pl_percent = (self.daily_pl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        logging.info(f"Daily Unrealized P/L: ${unrealized_pl:.2f} ({daily_pl_percent:.2f}%)")

        if self.positions:
            logging.info("\nCurrent Positions:")
            for symbol, position in self.positions.items():
                if position['shares'] > 0:
                    logging.info(f"{symbol}: {position['shares']} shares")
        else:
            logging.info("No open positions.")

        # Add timestamp to the log
        logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
