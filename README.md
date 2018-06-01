# Portfolio-management

## Overview
A small script I use to keep track of my stock portfolio. The data is obtained through two APIs: quandl [1] and alpha
vantage (thanks to Romel Torres for his python wrapper) [2] (following the severe restrictions in both google's and yahoo finance's APIs recently. For both a free API
key is available upon registering.

The idea was to create a period overview of my portfolio, performance and account. The focus is therefore not on
optimized performance. I.e. to account for backwards changes in price data the tables are constructed anew at every
request. Taking into account an inbuilt delay to the requests, building all overviews may take a minute or two for a
large numer of stocks.


## Features
- Take into account payments into / out of the account associated with the portfolio
- Take into account any fees arising
- Comparison to a freely chosen benchmark
- Conversion of foreign currency stock into a base currency
- Overviews, including money weighted returns ("IRR" - internal rate of return) for:
    - aggregated portfolio (yearly and over time)
    - account balance
    - individual stocks
- Handle splits

[INCLUDE SCREENSHOTS HERE]

For more details see the example notebook.

Requirements:
numpy, pandas, datetime, matplotlib, quandl, alpha vantage

### References
[1] https://www.quandl.com/
[2] https://www.alphavantage.co/documentation/, python wrapper: https://github.com/RomelTorres/alpha_vantage
