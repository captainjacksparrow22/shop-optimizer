from day_trading_advisor import DayTradingAdvisor
import time
from datetime import datetime
import sys
from pathlib import Path

class Logger:
    def __init__(self):
        self.log_dir = Path(r'C:\Users\Nathan\Documents\Python\shop-optimizer\stock-analyzer-2\logs')
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"day_trading_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
        self.terminal = sys.stdout
        self.log = open(self.log_file, 'a')

    def __init__(self):
        self.log_dir = Path(r'C:\Users\Nathan\Documents\Python\shop-optimizer\stock-analyzer-2\logs')
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"day_trading_log_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"
        self.terminal = sys.stdout
        self.log = open(self.log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        if hasattr(self, 'log'):
            self.log.close()

def main():
    # Set up logging
    sys.stdout = Logger()
    
    # Initialize the trading advisor
    advisor = DayTradingAdvisor(initial_balance=1000)
    symbol = 'BTC-USD'
    
    print(f"\n=== Day Trading Analysis for {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"Starting portfolio value: ${advisor.current_balance:.2f}")
    print(f"\nAnalyzing {symbol} on 5-minute timeframe...")
    
    try:
        # Analyze the symbol
        analysis = advisor.analyze_symbol(symbol)
        
        # Generate plot
        plot_file = advisor.plot_analysis(symbol)
        
        print(f"\n{'='*50}")
        print(f"BTC-USD Analysis at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"\nDECISION:")
        
        if analysis['recommendation'] == 'BUY' and analysis['probability'] > 0.65:
            print("[ACTION] >>> OPENING LONG POSITION <<<")
            print(f"Stop Loss will be set at: ${analysis['stop_loss']:.2f}")
        elif analysis['recommendation'] == 'SELL' and analysis['probability'] < 0.35:
            print("[ACTION] >>> CLOSING POSITION <<<")
        else:
            print("[ACTION] >>> NO ACTION NEEDED <<<")
        
        print("\nWHY?")
        if analysis['recommendation'] == 'BUY' and analysis['probability'] > 0.65:
            print("Bullish signals detected:")
            if analysis['trend_strength'].startswith('Strong Up'):
                print("• Strong upward trend")
            if analysis['volume_quality'] == 'Good':
                print("• Strong volume confirmation")
            print(f"• Risk level is {analysis['risk_level']}")
        elif analysis['recommendation'] == 'SELL' and analysis['probability'] < 0.35:
            print("Bearish signals detected:")
            if analysis['trend_strength'].startswith('Strong Down'):
                print("• Strong downward trend")
            if analysis['volume_quality'] == 'Poor':
                print("• Weak volume")
            print(f"• Risk level is {analysis['risk_level']}")
        else:
            print("Mixed or neutral signals:")
            print(f"• Current trend is {analysis['trend_strength']}")
            print(f"• Volume is {analysis['volume_quality']}")
            print(f"• Risk level is {analysis['risk_level']}")
            print("• Waiting for stronger signals")
        
        print(f"\nTechnical Analysis Plot saved to: {plot_file}")
        
        # Execute trade if recommended
        if analysis['recommendation'] in ['BUY', 'SELL']:
            trade = advisor.execute_trade(symbol, 
                                       analysis['recommendation'],
                                       analysis['current_price'],
                                       analysis['stop_loss'])
            
            if trade is not None:
                if 'investment' in trade:
                    print(f"\n[EXECUTED] Bought ${trade['investment']:.2f} of {symbol}")
                    print(f"[STOP LOSS] Set at: ${trade['stop_loss']:.2f}")
                elif 'amount' in trade:
                    print(f"\n[EXECUTED] Sold ${trade['amount']:.2f} of {symbol}")
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
    
    # Print current portfolio status
    print(f"\n{'='*50}")
    print("=== Current Status ===")
    print(f"Portfolio value: ${advisor.current_balance:.2f}")
    
    if advisor.portfolio:
        print("\nOpen Positions:")
        for sym, shares in advisor.portfolio.items():
            if shares > 0:
                value = shares * analysis['current_price']
                print(f"{sym}: {shares:.6f} shares (${value:.2f})")
    
    print(f"\nAnalysis and trade history saved in: {advisor.data_dir}")
    print(f"Log file saved in: {sys.stdout.log_file}")

if __name__ == "__main__":
    try:
        print("\nBTC-USD Day Trading Advisor Started...")
        print("Press Ctrl+C to stop the application\n")
        while True:
            main()
            print("\nWaiting 5 minutes for next analysis...")
            print("Press Ctrl+C to exit...")
            time.sleep(300)  # Wait 5 minutes before next analysis
    except KeyboardInterrupt:
        print("\n\nGracefully shutting down...")
        print("Trading session ended. Check logs for trade history.")
        sys.exit(0)
