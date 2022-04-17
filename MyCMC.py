import json, numpy
from matplotlib.pyplot import figure
from pandas import DataFrame, Series, Timedelta
from pandas import to_datetime as datetime
from IPython.display import display as print
from requests import Session
from pandas import concat
from ProgBar import ProgBar
from configparser import ConfigParser
from ccxt import binanceus as Binance

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class MyCMC:

    _CMCURL = "https://%s-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

    _dataColumns = ["time", "open", "high", "low", "close", "volume"]

    _jsonMapper = "BCMapper.json"

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __init__(self, test: bool = False):

        config = ConfigParser()
        config.read("config.ini")
        self.configBinance = dict(config["Binance"]) # Binance credentials from ini file.
        self.configCMC = dict(config["CoinMarketCap"]) # CMC credentials from ini file.
        self.headerCMC = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": self.configCMC[
                "apitest" if test else "apikey"]}
        self.CMCURL = self._CMCURL % ("sandbox" if test else "pro")
        with open(self._jsonMapper, "r") as file:
            self.mapper = json.load(file) # Dict/json that maps CMC coin tickers to Binance ones.
        self.dataCMC = self.dataBinance = None # These DataFrames will get filled in the future.
        self.requestCMC() # CoinMarketCap provides market cap ranking easily. Binance does not.

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def mapCB(self, cmcs: list[str]):
        """
        CoinMarketCap to Binance shortcut function. Leaves the unfound tickers as they are.\n
        Inputs: -> "cmcs" as list of ticker strs from CoinMarketCap.\n
        Outputs: -> "cmcs" as list of ticker strs from Binance.
        """

        for n, cmc in enumerate(cmcs):
            try: cmcs[n] = self.mapper[cmc]
            except: pass
        return cmcs

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def requestCMC(self, keepTop: int = None, saveCSV: bool = False) -> DataFrame:
        """
        Request a DataFrame of CoinMarketCap assets and their info, ranked by Market Cap (in Billions)\n
        Inputs:\n
            -> "keepTop" as the amount of top rows being kept in the rank. If None, it is stored as it is.\n
            -> "saveCSV" as a bool whether to save the DataFrame as CSV or not, just in case.\n
        Outputs: -> DataFrame with sorted top market caps (optional, else remains as instance attributes.)
        """

        params = {"start": 1, "limit": 5000, "convert": "USD"}
        # Request info through CoinMarketCap API:
        with Session() as session:
            session.headers.update(self.headerCMC)
            response = json.loads(session.get(self.CMCURL, params = params).text)
            # Session request will raise an exception and automatically stop the process at any HTTP error.

        # Keep only the quote info based in USD from the json response, and convert to DataFrame:
        quotes = DataFrame({ data["symbol"]: data["quote"]["USD"] for data in response["data"] })
        # Transpose: index as tickers, columns as variables. Everything except time should be float.
        quotes = quotes.transpose().astype(dict.fromkeys(quotes.index[:-1], float))
        quotes = quotes.sort_values("market_cap", ascending = False) # Sort; larger to smaller market cap.
        quotes["last_updated"] = datetime(quotes["last_updated"]) # Time column to datetime datatype.

        # Shorten column names, and discard non-necessary columns.
        keepColumns = {"market_cap": "m_cap", "market_cap_dominance": "mkt_%",
            "volume_24h": "v_24h", "price": "price", "last_updated": "last"}
        self.dataCMC = quotes[keepColumns.keys()].rename(columns = keepColumns)
        self.dataCMC["m_cap"] /= 1e9 # Easier to read market cap in billions in figures.
        if saveCSV: self.dataCMC.to_csv("QuotesCMC.csv") # Store data as CSV if wished.
        if keepTop: return self.getCMC(keepTop) # If top rows were input, output DataFrame.

    def getCMC(self, keepTop: int = 50) -> Series:
        """
        Fast retrieval of already stored top market cap cryptocurrencies.\n
        Inputs: -> "keepTop" as the amount of top rows desired in the rank.\n
        Outputs: -> DataFrame holding the top market cap assets.
        """

        return self.dataCMC["m_cap"].iloc[: keepTop]

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    @staticmethod
    def logRange(xmin: float, xmax: float, ints: list = [1, 2, 5]) -> numpy.ndarray:
        """
        Get a range of logarithmically spaced value intervals based on two end inputs. "ints" is a base
        list for the int coefficients of such spaces. Range will go from the previous "1 x 10^nMin" of
        "xmax", to the next "int x 10^nMax" of "xmax", "n" being the order of magnitude of each number.\n
        Some examples for different (xmin, xmax, ints):\n
        -> (123, 4567, [1, 2, 5]) => [100, 200, 500, 1000, 2000, 5000]\n
        -> (123, 5678, [1, 2, 5]) => [100, 200, 500, 1000, 2000, 5000, 10000]\n
        -> (123, 2000, [1, 2, 5]) => [100, 200, 400, 1000, 2000, 5000]\n
        -> (123, 2000, [1, 2, 4]) => [100, 200, 400, 1000, 2000, 4000]\n
        -> (123, 1999, [1, 2, 4]) => [100, 200, 400, 1000, 2000]\n
        -> ( 99, 5678, [1, 2, 5]) => [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]\n
        Serves as a log-grid-generating method.
        """
        logRange = numpy.log10([xmin, xmax])
        logRange = numpy.arange(*(logRange + [0, 2]))
        logRange = numpy.power(10, logRange.astype(int))
        logRange = numpy.outer(logRange, ints).reshape(-1)
        return logRange[: sum(logRange <= xmax) + 1]

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def plotCMC(self, figsize: tuple[float] = (15, 5), saveJPG: bool = False) -> figure:
        """
        Get a simple histogram showing the market caps of the top assets from larger to smaller.
        Amount of displayed assets depends on the figure width.\n
        Inputs:\n
            -> "figsize" as a 2-element tuple: (figure width, figure height).\n
            -> "saveJPG" as a bool whether wishing to store the figure as an in-folder JPG file or not.\n
        Output: -> "Figure" handle from Matplotlib.
        """

        # Number of assets/bars proportional to figure width.
        keepTop = figsize[0] * 50 / 15
        keepTop = int(numpy.ceil(keepTop / 10) * 10)
        keepTop = min(self.dataCMC.shape[0], keepTop)
        topCMC = self.getCMC(keepTop = keepTop)
        # Figure creation and customization.
        axes = topCMC.plot.bar(figsize = figsize, color = "orange")
        axes.set_yscale("log")
        axes.set_xticks(range(keepTop)) 
        xticks = topCMC.index.astype(str)
        axes.set_xticklabels(xticks, fontsize = 12, rotation = 90)
        yLogRange = self.logRange(topCMC.min(), topCMC.max())
        # Vertical tick grid generating.
        axes.set_yticks(yLogRange)
        axes.set_yticklabels(yLogRange, fontsize = 12)
        axes.set_ylabel("Market cap (Billion USD)", fontsize = 12, fontweight = "bold")
        axes.set_ylim(*yLogRange[[0, -1]]) # Fit grid to logRange.
        title = f"Top-50 CoinMarketCap\n"
        title += "‾" * int(len(title) * 1.2) # Title "underlining".
        axes.set_title(title, fontsize = 14, fontweight = "bold", va = "top")
        axes.grid(True, axis = "y", alpha = 1/4, color = "gray", lw = 3)
        axes.figure.set_tight_layout((0, 0, 1, 1)) # Make everything visible.
        if saveJPG: axes.figure.savefig(f"Top CMC {keepTop}.jpg", dpi = 100)
        return axes.figure

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def getBinance(self, timeFrame: str, since: str = None, keepTop: int = 50, saveCSV: bool = False):
        """
        Get "OHLC candles and volume" market data from the top-ranked market cap assets, and format
        it in a 2-level multi-indexed (asset, OHLCV) DataFrame.\n
        Inputs:\n
            -> "timeFrame" as the time "step" of each candle. Examples. "5m", "15m", "1h", "4h", etc.\n
            -> "since" as the timestamp (YYYY-MM-DD) of the oldest candle that the DataFrame must keep.\n
            -> "keepTop" as the amount of top rows being kept in the market cap rank.\n
            -> "saveCSV" as a bool whether it's wished to keep the candle DataFrame as CSV just in case.\n
        Outputs: None. DataFrame is stored as an instance attribute ("[instance].dataBinance").
        """
        binance = Binance(self.configBinance) # Create API session.
        if (since == None): # Make "since" an year ago if None.
            since = datetime("now", utc = True) - Timedelta(365, "day")
            since = since.strftime("%Y-%m-%d")
        args = {"limit": 50000, "timeframe": timeFrame}
        # Check if any CMC ticker is differently named in Binance.
        quotes = self.mapCB(self.getCMC(keepTop).index)
        # Create an empty dict to hold future candle data.
        self.dataBinance = dict.fromkeys(quotes)
        notFoundQuotes = list()

        progBar = ProgBar(quotes, 40, "Downloading from Binance")
        for quote in quotes: # Download each candle data as json.

            progBar.show()
            args["since"] = int(datetime(since).timestamp() * 1e3) + 1
            try: data = binance.fetch_ohlcv(quote + "/USD", **args)
            except Exception as e: notFoundQuotes.append(quote) ; continue
            if (len(data) == 0): notFoundQuotes.append(quote) ; continue

            while True: # Recent rows may be missing because of API limit.
                args["since"] = data[-1][0] + 60000 # Next row datetime. At least 1 min from prev.
                try: data2 = binance.fetch_ohlcv(quote + "/USD", **args) # Download from that point.
                except Exception as e: break # Stop if no more to download:
                if (len(data2) == 0) or (data[-1][0] == data2[-1][0]): break
                else: data += data2 # Complete dataframe with new values.

            data = DataFrame(data, columns = self._dataColumns) # json to DataFrame.
            data.set_index("time", inplace = True) # Keep candle timestamps as row indexes.
            data.index = datetime(data.index, unit = "ms") # Strings to datetime data types.
            data["volume"] = data["volume"].fillna(0) # (Explained below @ line 194)
            data["close"] = data["close"].ffill().bfill() # (Explained below @ line 196)
            data = data.bfill(axis = "columns") # (Explained below @ line 199)
            self.dataBinance[quote] = data.astype(float) # All data should be float numbers.

        if notFoundQuotes: print("Warning! Quotes not found in Binance:", str(notFoundQuotes))

        self.dataBinance = concat(self.dataBinance, axis = "columns")
        self.dataBinance = self.dataBinance.swaplevel(axis = "columns")
        # Candles with no activity or price change should have zero volume.
        self.dataBinance["volume"] = self.dataBinance["volume"].fillna(0)
        # Close price of zero volume candles should be equal to the last close.
        self.dataBinance["close"] = self.dataBinance["close"].ffill().bfill()
        self.dataBinance = self.dataBinance.swaplevel(axis = "columns")
        # Any price (OHL) of zero volume candles should be equal to its close.
        self.dataBinance = self.dataBinance.bfill(axis = "columns")
        quotes = self.dataBinance.columns.levels[0] # Columns multi-index: (quotes, OHLCV)
        if saveCSV: # Filename example: "Top-30 CMC 4h, 2022.01.01-2022.04.14.csv"
            t1, t2 = self.dataBinance.index[[0, -1]].strftime("%Y.%m.%d")
            title = "Top-%d CMC %s, %s-%s" % (keepTop, args["timeframe"], t1, t2)
            self.dataBinance.reset_index().to_csv(title + ".csv", index = False)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

if (__name__ == "__main__"):

    myCMC = MyCMC()
    myCMC.requestCMC(keepTop = 50)
    myCMC.plotCMC(saveJPG = True)
    myCMC.getBinance(timeFrame = "4h")
    print(myCMC.dataBinance)