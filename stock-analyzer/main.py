from trading_advisor import TradingAdvisor

def main():
    """
    Main function to run the swing trading advisor.
    Initializes the advisor and runs the analysis for a predefined list of symbols.
    """
    # Initialize the trading advisor with a $1000 starting balance
    advisor = TradingAdvisor(initial_balance=1000)
    
    # Focus on BTC-USD and leveraged NASDAQ ETFs
    symbols = ['BTC-USD', 'TQQQ', 'SQQQ']
    
    # The analyze method now handles the entire process:
    # - Logging setup
    # - Looping through symbols
    # - Fetching data and analyzing
    # - Executing trades
    # - Logging the summary
    advisor.analyze(symbols)

if __name__ == "__main__":
    main()
