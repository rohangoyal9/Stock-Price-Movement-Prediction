# Stock Price Movement Prediction
The aim of this project is to investigate the performance of various machine learning models to predict stock market movements based on historical time series data and news article sentiment collected using APIs and web scraping.  The basic principle would be to buy low and sell high, but the complexity arises in knowing when to buy and sell a stock.
Four types of analysis exist to forecast the markets:-
1) Fundamental
2) Technical 
3) Quantitative
4) Sentiment

Each type of analysis has its own underlying principles, tools, techniques and strategies, and it is likely that understanding the intuition of each and combining complementary approaches is more optimal than relying solely on one. Forecasting strategies will be developed based on predictions and backtested against a benchmark.

The NIFTY 50 is a benchmark Indian stock market index that represents the weighted average of 50 of the largest Indian companies listed on the National Stock Exchange. This study focuses on data from five of the top NIFTY 50 companies (Reliance Industries, Tata consultancy services, ONGC, Infosys, ITC Ltd.) representing a range of sectors.

## Data Analysis:
The Yahoo Finance API is used to download stock data for the opening price (Open), the highest and lowest price the stock traded at (High, Low), closing price (Close), and the number of stocks traded (Volume) and Adjusted Close. To give a more accurate reflection of the true value of the stock and present a coherent picture of returns, the Adjusted Close price is typically used for prediction purposes. It is an estimate incorporating corporate actions, including splits and dividends, which gives a more accurate representation of the true value of the stock.

![image](https://www.linkpicture.com/q/Screenshot-2022-07-01-at-2.19.21-PM.png)

Data is transformed to calculate and visualize returns, and covariance and correlation matrices show the strength and direction of the relationship between stocks' returns. These observations could be used to select a portfolio of stocks that complement each other in terms of price movement.

## Technical Analysis:
Technical analysis is the use of charts and technical indicators to identify trading signals and price patterns. Various technical strategies are investigated using the most common leading and lagging trend, momentum, volatility, and volume indicators including Moving Averages, Moving Average Convergence Divergence (MACD), Stochastic Oscillator, Relative Strength Index (RSI), Money Flow Index (MFI), Rate of Change (ROC), Bollinger Bands, and On-Balance-Volume (OBV).

## Time Series Analysis:
A time series is a series of data points ordered in time and is an important factor in predicting stock market trends. In time series forecasting models, time is the independent variable and the goal is to predict future values based on previously observed values.
Stock prices are often non-stationary and may contain trends or volatility but different transformations can be applied to turn the time series into a stationary process so that it can be modeled.

#### 1. Facebook Prophet:
Prophet is an open-source library published by Facebook in 2017 which is built upon scikit-learn time series modeling and automatically detects changes in trends by selecting changepoints from the data. It is an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects, and includes automatic detection for all values.

![image](https://www.linkpicture.com/q/Screenshot-2022-07-01-at-2.22.52-PM.png)

#### 2. Long Short Term Memory (LSTM):
Recurrent Neural Network (RNN) model such as Long Short-Term Memory (LSTM) is explored and various machine learning and deep learning models are created, trained, tested, and optimized.
Five years of Reliance Industries' historical stock data is used to predict Adjusted Close prices by building a multi-layer LSTM Recurrent Neural Network model. The ability to store information over a period of time is useful when dealing with time-series data.

![image](https://www.linkpicture.com/q/Screenshot-2022-07-01-at-2.24.57-PM.png)

## Sentiment Analysis:
We are  fetching daily stock market data of Reliance from yfinance api and its daily news articles from mediastack api and creating a derived dataset which contains news sentiment. 
For sentiment analysis ,  huggingface FinBERT model has been used. FinBERT is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification.

### Python Libraries Used:
* Numpy
* Pandas
* Matplotlib
* Mplfinance
* Seaborn
* Plotly
* SciPy
* Statsmodels
* Scikit-learn
* Keras
* TensorFlow
* Yfinance
* Beautiful Soup
* Selenium
* TextBlob
* SpaCy
* Gensim
* BERT
* Hugging Face
* PyTorch



