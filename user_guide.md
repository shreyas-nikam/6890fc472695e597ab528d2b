id: 6890fc472695e597ab528d2b_user_guide
summary: Lab 2.1: PD Models - Development User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Interactive Stock Analysis with Streamlit

This application provides an interactive platform for analyzing stock data. It allows users to visualize historical stock prices, explore moving averages, and compare the performance of different stocks. This tool is valuable for anyone interested in gaining insights into stock market trends and making informed investment decisions. By using interactive charts and simple controls, the application makes complex data accessible and easy to understand.

## Getting Started: Understanding the Interface
Duration: 00:02

Upon launching the application, you'll notice a clean and intuitive interface. The sidebar on the left contains the controls for selecting stocks and customizing the analysis. The main area displays interactive charts visualizing the stock data. Take a moment to familiarize yourself with the different sections before diving into the analysis.

## Selecting Stocks for Analysis
Duration: 00:03

The first step is to choose the stocks you want to analyze. In the sidebar, you'll find a multi-select box labeled "Select Stocks". You can select one or more stocks from the available options. As you select stocks, the application automatically fetches their historical data and displays it in the main chart. You can select multiple stocks to compare their performance side-by-side.

## Visualizing Stock Prices
Duration: 00:05

The primary chart displays the historical stock prices for the selected stocks. The x-axis represents the date, and the y-axis represents the stock price. Each stock is represented by a different colored line. Hovering over the chart will reveal the exact date and price at that point. You can zoom in on specific time periods to examine price movements in more detail. This visualization allows you to quickly identify trends, patterns, and significant price changes.

## Exploring Moving Averages
Duration: 00:05

Moving averages are a valuable tool for smoothing out price data and identifying trends. The application allows you to add moving averages to the chart. In the sidebar, you'll find a number input labeled "Moving Average Window." Enter the number of days you want to use for the moving average calculation.  The application will then calculate and display the moving average for each selected stock. Comparing the stock price to its moving average can help you identify potential buy and sell signals. A moving average is calculated as:

$$MA_t = \frac{P_t + P_{t-1} + ... + P_{t-n+1}}{n}$$

where $MA_t$ is the moving average at time $t$, $P_t$ is the price at time $t$, and $n$ is the window size.

<aside class="positive">
Experiment with different moving average windows to see how they affect the smoothness of the curve and the accuracy of trend identification.
</aside>

## Comparing Stock Performance
Duration: 00:05

One of the key features of this application is the ability to compare the performance of different stocks. By selecting multiple stocks, you can visualize their price movements on the same chart. This allows you to quickly identify which stocks have performed better than others over a given period. You can also compare the moving averages of different stocks to see which ones have stronger trends.

## Adjusting the Time Period
Duration: 00:03

You can adjust the time period for the analysis using the date input fields in the sidebar. This allows you to focus on specific periods of interest, such as the last year, the last month, or a custom date range. By adjusting the time period, you can gain a more detailed understanding of how stocks have performed under different market conditions.

## Understanding Log Scale
Duration: 00:02

The application also features a checkbox to toggle a Log Scale. When enabled, the Y axis (price) will use a logarithmic scale. Logarithmic scales are useful when visualizing data that spans several orders of magnitude, which is common in stock prices over extended periods. This allows the user to clearly see percentage changes rather than absolute changes.

<aside class="negative">
Be mindful of the time period when comparing stock performance, as different stocks may have been affected by different events during different periods.
</aside>

## Conclusion
Duration: 00:01

This interactive stock analysis application provides a powerful tool for visualizing and comparing stock data. By using the interactive controls, you can explore historical prices, moving averages, and relative performance of different stocks. This can help you gain valuable insights into stock market trends and make more informed investment decisions. Remember that this application is for educational purposes only and should not be used as the sole basis for making investment decisions.
