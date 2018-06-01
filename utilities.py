import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import time
# import fix_yahoo_finance as yf
# import pandas_datareader as pdr
import quandl
from alpha_vantage.timeseries import TimeSeries
from utilities import *

QUANDL_KEY = open('quandl_API.txt', 'r').readline()
VANTAGE_KEY = open('alpha_vantage_API.txt', 'r').readline()
ts = TimeSeries(key=VANTAGE_KEY, output_format='pandas')

def download_price(ticker, start, end=datetime.today(), quandl_API=False):
    '''
    :param ticker:
    :param start:
    :param end:
    :return: DataFrame with at least column "Price"
    '''
    if quandl_API:
        data = quandl.get(ticker, start_date=start, end_date=end, api_key=QUANDL_KEY)
    else:
        try:
            time.sleep(1)
            data, meta_data = ts.get_daily_adjusted(ticker, outputsize='full')
            data = data.rename(columns={"4. close": "Price"})
            data.index = [datetime.strptime(i, "%Y-%m-%d") for i in data.index.tolist()]
            data = data.loc[start:]
        except ValueError:
            data = quandl.get(ticker, start_date=start, end_date=end, api_key=QUANDL_KEY)
            quandl_API = True

    if "Value" in data.columns:
        data = data.rename(columns={"Value": "Price"})

    return data, quandl_API


class Portfolio:

    def __init__(self, bm_key, bm_series, start_date):
        self.stocks = []
        self.start_date = datetime.strptime(start_date, '%d/%m/%Y')
        self.bm_key = bm_key
        self.bm_series = bm_series

        cols = ["Return absolute", "Invested capital", "Benchmark"]
        self.overview = pd.DataFrame(data=[len(cols) * [0.]],
                                     columns=cols,
                                     index=pd.DatetimeIndex(start=self.start_date, end=datetime.today(), freq='D',
                                                            name='Date'))

        cols = ["Payments", "Fees"]
        self.balance = pd.DataFrame(data=[len(cols) * [0.]],
                                    columns=cols,
                                    index=pd.DatetimeIndex(start=self.start_date, end=datetime.today(), freq='D',
                                                           name='Date'))

        self.yearly_returns = None
        self.yearly_returns_bm = None
        cols = ["Currently held", "Prev. sold positions", "Currently invested capital", "Current value",
                "Return absolute", "IRR", "IRR-BM", "Dividends"]
        self.stock_stats = pd.DataFrame(data=[len(cols)*[0.]], columns=cols, index=["Portfolio"])

    @property
    def stock_list(self):
        return [s.ticker for s in self.stocks]

    def get_stock(self, name):
        stock = [s for s in self.stocks if (s.ticker == name) or (s.display_name == name)]

        if not stock:
            print(name + " not in portfolio")
            return None
        else:
            return stock[0]

    def buy(self, ticker, date, number, price_per_share, commission, currency=None, old_ticker=None, display_name=None):
        '''date: day/month/year'''
        s = self.get_stock(ticker)
        if s:
            s._buy(date, number, price_per_share, commission)
        else:
            self.stocks.append(Stock(ticker, display_name, date, number, price_per_share, commission, currency, old_ticker))
        print("Bought {} {}".format(number, ticker))

    def sell(self, ticker, date, number, price_per_share, commission):
        '''date: day/month/year'''
        s = self.get_stock(ticker)
        if s:
            s._sell(date, number, price_per_share, commission)

        print("Sold {} {}".format(number, ticker))

    def dividend_payment(self, ticker, date, total_div):
        '''date: day/month/year'''
        s = self.get_stock(ticker)
        if s:
            s._dividend_payment(date, total_div)

    def split(self, ticker, date, ratio):
        '''date: day/month/year'''
        s = self.get_stock(ticker)
        if s:
            s._split(date, ratio)

    def pay_in(self, date, amount):
        self.balance.loc[datetime.strptime(date, '%d/%m/%Y'), "Payments"] = amount

    def pay_out(self, date, amount):
        self.balance.loc[datetime.strptime(date, '%d/%m/%Y'), "Payments"] = -amount

    def pf_fee(self, date, fee):
        self.balance.loc[datetime.strptime(date, '%d/%m/%Y'), "Fees"] = fee

    def update_returns(self):
        [s.update_returns() for s in self.stocks]

    def create_overview(self):
        self.overview["Benchmark"] = download_price(self.bm_key, start=self.start_date)[0][self.bm_series]

        self.update_returns()

        cols = ["Return absolute", "Invested capital"]
        for s in self.stocks:
            self.overview[cols] = self.overview[cols].add(s.stats[cols].fillna(0), fill_value=0)

        tmp = self.balance.sort_index().join(self.overview, how='outer')["Fees"].fillna(0).cumsum()
        self.overview["Return absolute"] = self.overview["Return absolute"].add(tmp)

        # fill up non-trading days
        self.overview[self.overview==0] = np.nan
        self.overview = self.overview.ffill()

        self.plot_pf()
        self.create_stock_overview()
        self.create_balance_overview()

        return self.overview

    def plot_pf(self):
        # absolute values
        f, axes = plt.subplots(1, 2, figsize=(16, 6))
        self.overview[["Return absolute", "Invested capital"]].plot(ax=axes[0])
        ax2 = axes[0].twinx()
        (self.overview["Benchmark"] / self.overview.loc[self.overview["Benchmark"].first_valid_index(), "Benchmark"]).plot(ax=ax2, color='g')
        ax2.set_ylabel('Benchmark', color='g')
        axes[0].set_title("Portfolio stats")

        tmp = self.overview["Return absolute"].resample("Y").last().diff(periods=1)
        self.yearly_returns = tmp / self.overview["Invested capital"].resample("Y").mean()
        self.yearly_returns_bm = self.overview["Benchmark"].resample("Y").last().pct_change()
        df = pd.concat([self.yearly_returns, self.yearly_returns_bm], axis=1)
        df.index = df.index.year
        df.columns = ["pf", "Benchmark"]
        df.plot.bar(ax=axes[1])
        axes[1].set_title("Yearly returns over mean invested capital")
        plt.show()

        n = int(np.ceil(len(self.stocks) / 2))
        f, axes = plt.subplots(n, 2, figsize=(16, n*3.5))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(self.stocks):
                break
            else:
                ax2 = ax.twinx()
                p1, = ax.plot(self.stocks[i].stats["Return absolute"], label="Return absolute")
                p2, = ax.plot(self.stocks[i].stats["Invested capital"], label="Invested capital")
                # ax.legend()
                p3, = ax2.plot(self.stocks[i].stats["Price"], color='g', ls=":", label="Price")
                lns = [p1, p2, p3]
                if self.stocks[i].currency:
                    ax22 = ax.twinx()
                    p4, = ax22.plot(self.stocks[i].stats["Currency"], color='k', ls=":", label=self.stocks[i].currency)
                    lns.append(p4)
                    ax22.spines['right'].set_position(('outward', 60))
                    ax22.xaxis.set_ticks([])
                    ax22.set_ylabel(self.stocks[i].currency, color='k')

                    # ax.set_ylabel('Price', color='b')

                ax.legend(handles=lns, loc='best')
                ax2.set_ylabel('Price', color='g')


                ax.set_title(self.stocks[i].display_name)

        f.tight_layout()
        plt.show()

    def create_stock_overview(self):
        irr = -self.balance["Fees"]
        for s in self.stocks:
            self.stock_stats.loc[s.display_name] = [s.total_number,
                                              s.events["Sold again"].sum(),
                                              s.stats["Invested capital"][-1],
                                              s.stats["Value over time"][-1],
                                              s.stats["Return absolute"][-1],
                                              s.stats["IRR"][-1],
                                              np.nan,
                                              s.events["Dividend"].sum()]

            irr = irr.add(s.stats["IRR-series, current price"], fill_value=0)

        bm      =  self.overview["Benchmark"].resample("Y").last().diff(periods=1)
        bm[0]   = -self.overview["Benchmark"].resample("Y").first()[0]
        bm[-1]  += self.overview["Benchmark"].resample("Y").last().iloc[-1]
        self.stock_stats.loc["Portfolio", "IRR-BM"] = np.irr(bm)

        self.stock_stats.loc["Portfolio", ["Currently held", "Prev. sold positions", "Currently invested capital", "Current value", "Dividends"]] = self.stock_stats[["Currently held", "Prev. sold positions", "Currently invested capital", "Current value", "Dividends"]].sum()
        self.stock_stats.loc["Portfolio", "Return absolute"] = self.overview["Return absolute"][-1]
        self.stock_stats.loc["Portfolio", "IRR"] = np.irr(irr.resample("Y").sum())

        display(self.stock_stats)

    def create_balance_overview(self):
        # self.balance = self.balance.sort_index().fillna(0)
        self.balance["Cash flow"]               = self.balance["Payments"] - self.balance["Fees"]
        self.balance["PF value, selling today"] = self.balance["Payments"] - self.balance["Fees"]
        for s in self.stocks:
            self.balance["Cash flow"] = (self.balance["Cash flow"]
                                         .add(s.stats["IRR-series, realized"], fill_value=0)
                                         )
            self.balance["PF value, selling today"] = (self.balance["PF value, selling today"]
                                                       .add(s.stats["IRR-series, current price"], fill_value=0)
                                                       )

        tmp = (self.balance[["Cash flow", "Payments", "PF value, selling today"]].cumsum()
               .join(self.overview["Invested capital"])
               )
        tmp["PF value, selling today"] = tmp["PF value, selling today"].add(tmp["Invested capital"])
        tmp.plot(title="Portfolio balance", figsize=(10, 6))
        print("PF as of today:")
        display(self.balance.sum())


class Stock:

    def __init__(self, ticker, display_name, date, number, price_per_share, commission, currency=None, old_ticker=None):
        '''
        :param ticker:
        :param date:
        :param number:
        :param price_per_share:
        :param commission:
        :param currency: as CHF into foreign
        :param old_ticker:
        '''
        self.ticker = ticker
        if not display_name:
            self.display_name = ticker
        else:
            self.display_name = display_name
        self.currency = currency
        self.old_ticker = old_ticker
        self.open_position = True
        self.splits = []
        self.quandl_API = False

        self.events = pd.DataFrame(data=[[number,
                                          price_per_share,
                                          number * price_per_share,
                                          commission, 0.,
                                          0.,
                                          number * price_per_share]],
                                   columns=["Number", "Price per share", "Total Price", "Commission", "Dividend", "Sold again", "Invested capital"],
                                   index=pd.DatetimeIndex(start=date, freq='D', periods=1, name='Date'))

    def _buy(self, date, number, price_per_share, commission):
        '''date: day/month/year'''
        self.open_position = True
        self.events.loc[datetime.strptime(date, '%d/%m/%Y')] = [number,
                                                                price_per_share,
                                                                number * price_per_share,
                                                                commission,
                                                                0,
                                                                0,
                                                                number * price_per_share]

    def _sell(self, date, number, price_per_share, commission):
        '''date: day/month/year'''
        if self.total_number < number:
            print("Selling more shares than you have in your portfolio! ")
            raise ValueError

        # reduce "invested capital" by price of first still "unsold" position
        n, p, i = 0, 0, 0
        while n < number:
            row = self.events.iloc[i]
            if row["Sold again"] < row["Number"]:
                self.events.loc[self.events.index[i], "Sold again"] = min(row["Number"], number)
                n += self.events.loc[self.events.index[i], "Sold again"]
                p += self.events.loc[self.events.index[i], "Sold again"] * row["Price per share"]

            i += 1

        self.events.loc[datetime.strptime(date, '%d/%m/%Y')] = [-number,
                                                                price_per_share,
                                                                -number * price_per_share,
                                                                commission,
                                                                0,
                                                                0,
                                                                -p]

        if self.total_number == 0:
            self.update_returns()
            self.open_position = False

    def _dividend_payment(self, date, total_div):
        self.events.loc[datetime.strptime(date, '%d/%m/%Y'), "Dividend"] = total_div

    def _split(self, date, ratio):
        self.events[["Number", "Sold again"]] = self.events[["Number", "Sold again"]] * ratio
        self.events["Price per share"] = self.events["Price per share"] / ratio

        self.splits.append((date, ratio))

    @property
    def total_number(self):
        return self.events["Number"].sum()

    def get_event(self, date):
        return self.events.loc[datetime.strptime(date, '%d/%m/%Y')]

    def plot_price(self, start=None, end=datetime.today()):
        if start is None:
            start = self.events.index[0]

        data, _ = download_price(self.ticker, start, end, self.quandl_API)
        data.Price.plot(title=self.display_name)
        plt.show()

    def update_returns(self):
        if self.open_position:
            data, quandl_API = download_price(self.ticker, self.events.index[0], datetime.today(), self.quandl_API)
            self.quandl_API = quandl_API
            if self.old_ticker:
                data_old, _ = download_price(self.old_ticker, self.events.index[0], data.index[0] - timedelta(days=1), self.quandl_API)
                data = data_old.append(data)

            data = data.join(self.events, how='outer')

            cols = ["Price", "Value over time", "Return absolute", "Invested capital"]
            self.stats = pd.DataFrame(data=[len(cols) * [0.]],
                                      columns=cols,
                                      index=pd.DatetimeIndex(start=self.events.index[0],
                                                             end=datetime.today(),
                                                             freq='D',
                                                             name='Date'))

            # convert all into base currency
            if self.currency:
                data["Currency"] = 1 / download_price(self.currency, start=self.events.index[0], quandl_API=True)[0]["Price"].ffill()
            else:
                data["Currency"] = pd.Series(1., index=data["Price"].index)
            data[["Dividend", "Total Price", "Invested capital"]] = (data[["Dividend", "Total Price", "Invested capital"]]
                                                                     .multiply(data["Currency"], axis="index")
                                                                     )

            self.stats["Price"] = data["Price"].ffill() * data["Currency"].ffill()
            self.stats["Currency"] = data["Currency"].ffill()

            # Quandl does not adjust prices before split
            if self.quandl_API:
                for i in range(len(self.splits)):
                    if i == 0:
                        self.stats.loc[:self.splits[i][0], "Price"] = self.stats.loc[:self.splits[i][0], "Price"] / self.splits[i][1]
                    else:
                        self.stats.loc[self.splits[i-1][0]:self.splits[i][0], "Price"] = self.stats.loc[self.splits[i-1][0]:self.splits[i][0], "Price"] / self.splits[i][1]

            data[data.isna()==True] = 0

            self.stats["Value over time"]  = (data["Number"].cumsum() * self.stats["Price"]
                                              + data["Dividend"].cumsum()
                                              - data["Commission"].cumsum())
            self.stats["Return absolute"]  = (data["Number"].cumsum() * self.stats["Price"]
                                              + data["Dividend"].cumsum()
                                              - data["Commission"].cumsum()
                                              - data["Total Price"].cumsum())
            self.stats["Invested capital"] = data["Invested capital"].cumsum()
            self.stats = self.stats.ffill()

            self.stats["IRR-series, realized"] = (-data["Total Price"]
                                                  + data["Dividend"]
                                                  - data["Commission"])
            self.stats["IRR-series, realized"] = self.stats["IRR-series, realized"].fillna(0)
            self.stats["IRR-series, current price"] = self.stats["IRR-series, realized"].copy()
            self.stats["IRR-series, current price"].iloc[-1] = (self.stats["IRR-series, current price"].iloc[-1]
                                                                + self.stats["Price"].iloc[-1] * self.total_number)
            self.stats["IRR"] = np.irr(self.stats["IRR-series, current price"].resample("Y").sum())
