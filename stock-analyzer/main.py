from trading_advisor import TradingAdvisor
import time
from datetime import datetime
import sys
import os
from pathlib import Path

class Logger:
    def __init__(self):
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"trading_log_{datetime.now().strftime('%Y-%m-%d')}.txt"
        self.terminal = sys.stdout
        self.log = open(self.log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Set up logging
    sys.stdout = Logger()
    
    # Initialize the trading advisor with $1000 starting balance
    advisor = TradingAdvisor(initial_balance=1000)
    
    # Focus on BTC-USD and leveraged NASDAQ ETFs
    symbols = ['BTC-USD', 'TQQQ', 'SQQQ']
    
    print(f"\n=== Swing Trading Analysis for {datetime.now().strftime('%Y-%m-%d')} ===")
    print(f"Starting portfolio value: ${advisor.get_portfolio_value():.2f}")
    print("\nAnalyzing market conditions...")
    
    for symbol in symbols:
        try:
            # Analyze each symbol
            analysis = advisor.analyze_symbol(symbol)
            
            # Generate plot
            plot_file = advisor.plot_analysis(symbol)
            
            print(f"\n{'='*50}")
            print(f"{symbol} Swing Trade Analysis:")
            print(f"Current Price: ${analysis['current_price']:.2f}")
            print(f"Buy Probability: {analysis['buy_probability']:.2%}")
            print(f"Recommendation: {analysis['recommendation']}")
            print(f"Risk Level: {analysis['risk_level']}")
            
            # Additional swing trading metrics
            print("\nKey Metrics:")
            print(f"Trend Strength: {analysis['trend_strength']}")
            print(f"Volume Analysis: {analysis['volume_signal']}")
            print(f"Volatility Status: {analysis['volatility_status']}")
            
            # Print swing-specific advice
            print(f"\nSwing Trade Advice:")
            print(analysis['swing_advice'])
            
            print(f"\nTechnical Analysis Plot saved to: {plot_file}")
            
            # Paper trading logic based on recommendations and available balance
            if analysis['recommendation'] == 'BUY' and advisor.current_balance > 100:
                # More aggressive position sizing for strong signals
                confidence_factor = analysis['buy_probability']
                max_position = min(advisor.current_balance * 0.4, advisor.current_balance * confidence_factor)
                
                # Adjust position size based on volatility
                if analysis['risk_level'] == 'HIGH':
                    investment = max_position * 0.5  # Reduce position size for high risk
                else:
                    investment = max_position
                
                trade = advisor.paper_trade(symbol, 'buy', investment)
                print(f"\nExecuted paper trade: Bought ${investment:.2f} of {symbol}")
                print(f"Trade Reasoning: {analysis['trade_reasoning']}")
            
            elif analysis['recommendation'] == 'SELL' and symbol in advisor.portfolio:
                # Sell entire position if sell signal
                current_shares = advisor.portfolio.get(symbol, 0)
                if current_shares > 0:
                    sell_amount = current_shares * analysis['current_price']
                    trade = advisor.paper_trade(symbol, 'sell', sell_amount)
                    print(f"\nExecuted paper trade: Sold ${sell_amount:.2f} of {symbol}")
                    print(f"Trade Reasoning: {analysis['trade_reasoning']}")
            
            time.sleep(1)  # Avoid hitting API rate limits
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    # Print end of day summary
    print(f"\n{'='*50}")
    print("=== End of Day Summary ===")
    print(f"Current portfolio value: ${advisor.get_portfolio_value():.2f}")
    profit_loss = advisor.get_portfolio_value() - advisor.initial_balance
    print(f"Daily P/L: ${profit_loss:.2f} ({(profit_loss/advisor.initial_balance)*100:.2f}%)")
    print(f"\nCurrent Positions:")
    for symbol, shares in advisor.portfolio.items():
        if shares > 0:
            value = shares * advisor.get_current_price(symbol)
            print(f"{symbol}: {shares:.4f} shares (${value:.2f})")
    print("\nTrade history and analysis data saved in the 'data' directory")
    print(f"Today's analysis log saved in: {sys.stdout.log_file}")

if __name__ == "__main__":
    main()
