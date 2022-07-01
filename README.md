# Stock-Price-Movement-Prediction

The aim of the project is to investigate the performance of various machine learning models to predict stock market movements based on historical time series data and news article sentiment collected using APIs and web scraping. The basic principle would be to buy low and sell high, but the complexity arises in knowing when to buy and sell a stock.
Four types of analysis exist to forecast the markets:-
Fundamental
Technical 
Quantitative
Sentiment
Each type of analysis has its own underlying principles, tools, techniques and strategies, and it is likely that understanding the intuition of each and combining complementary approaches is more optimal than relying solely on one. Forecasting strategies will be developed based on predictions and backtested against a benchmark.
The NIFTY 50 is a benchmark Indian stock market index that represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange. This study focuses on data from five of the top NIFTY 50 companies (Reliance Industries, Tata consultancy services, ONGC, Infosys, ITC Ltd.) representing a range of sectors.
Data Analysis:
The Yahoo Finance API is used to download stock data for the opening price (Open), the highest and lowest price the stock traded at (High, Low), closing price (Close), and the number of stocks traded (Volume) and Adjusted Close. To give a more accurate reflection of the true value of the stock and present a coherent picture of returns, the Adjusted Close price is typically used for prediction purposes. It is an estimate incorporating corporate actions, including splits and dividends, which gives a more accurate representation of the true value of the stock.
