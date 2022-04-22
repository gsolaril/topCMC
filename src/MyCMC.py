import json, numpy
from matplotlib import figure
from matplotlib.pyplot import figure
from pandas import DataFrame, Series, Timedelta
from pandas import to_datetime as datetime
from IPython.display import display as print
from requests import Session
from pandas import concat
from .ProgBar import ProgBar
from configparser import ConfigParser
from ccxt import binanceus as Binance

class TickRange:

    __defaultLinBases, __defaultLogBases = numpy.array([1, 2, 4, 5, 10]), numpy.array([1, 2, 5])

    @classmethod
    def lin(cls, xMin: float, xMax: float, maxLength: int = 50) -> numpy.ndarray:
        """
        Get a range of linearly spaced values based on two end inputs. Space (tick) will be smaller
        or equal to the distance between both inputs, divided by "`maxLength`". Then, resulting extreme
        values will be a result of "`xMin`" and "`xMax`" rounded down and up respectively towards the
        calculated tick. The significand of such tick will be a common divisor of powers of 10 =>
        `[1, 2, 4, 5, 10]`.\n
        The resulting array:\n
        -> ...will never be larger than "`maxLength`".\n
        -> ...will always include zero if it has both positive and negative values.\n
        Some examples for different (`xMin, xMax, maxLength`):\n
        -> ( `1.0`, `5.0`, `10`) => tick = 0.5 => `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`\n
        -> ( `1.0`, `5.0`, `11`) => tick = 0.4 => `[1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]`\n
        -> (`-1.0`, `5.0`, `11`) => tick = 1.0 => `[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`\n
        -> (`-1.0`, `2.0`, `10`) => tick = 0.4 => `[-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0]`\n
        -> (`-100`, `200`, `10`) => tick = 40 => `[-120, -80, -40, 0, 40, 80, 120, 160, 200]`\n
        -> (`-100`, `201`, `10`) => tick = 40 => `[-120, -80, -40, 0, 40, 80, 120, 160, 200, 240]`\n
        -> (`-100`, `100`, `20`) => tick = 20 => `[-100, -80, -40, -20, 0, 20, 40, 60, 80, 100]`\n
        -> (`-100`, `100`, `21`) => tick = 10 => `[-100, -90, -80..., -20, -10, 0, 10, 20, ...80, 90, 100]`\n
        Serves as a linear grid-generating method. Values can be any real number.
        """
        tick = abs(xMax - xMin) / maxLength
        ints = numpy.array(cls.__defaultLinBases)
        exp10 = 10.0 ** numpy.floor(numpy.log10(tick))
        tick = ints[tick / exp10 < ints][0] * exp10
        xMin = int(numpy.floor(xMin / tick)) * tick
        xMax = int(numpy.ceil(xMax / tick)) * tick
        n = int((xMax - xMin) / tick) + 1
        return numpy.linspace(xMin, xMax, n)

    @classmethod
    def log(cls, xMin: float, xMax: float, ints: list = None) -> numpy.ndarray:
        """
        Get a range of logarithmically spaced value intervals based on two end inputs. "`ints`" is a base
        list for the int coefficients of such spaces. Default is => `[1, 2, 5]`. Range will go from the
        previous "1 x 10^nMin" of "`xMax`", to the next "int x 10^nMax" of "`xMax`", Each "n" being the
        order of magnitude of its number.\n
        Some examples for different (`xMin, xMax, ints`):\n
        -> (`123`, `4567`, `[1, 2, 5]`) => nMin = 2, nMax = 3 => `[100, 200, 500, 1000, 2000, 5000]`\n
        -> (`123`, `5678`, `[1, 2, 5]`) => nMin = 2, nMax = 4 => `[100, 200, 500, 1000, 2000, 5000, 10000]`\n
        -> (`123`, `2000`, `[1, 2, 5]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000, 5000]`\n
        -> (`123`, `2000`, `[1, 2, 4]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000, 4000]`\n
        -> (`123`, `1999`, `[1, 2, 4]`) => nMin = 2, nMax = 3 => `[100, 200, 400, 1000, 2000]`\n
        -> ( `99`, `5678`, `[1, 2, 5]`) => nMin = 1, nMax = 3 => `[10, 20, 50, 100..., 2000, 5000, 10000]`\n
        Serves as a logarithmic grid-generating method. NOTE: Values CANNOT be negative.
        """
        if isinstance(ints, type(None)):
            ints = cls.__defaultLogBases
        ticks = numpy.log10([xMin, xMax])
        ticks = numpy.arange(*(ticks + [0, 2]))
        ticks = numpy.power(10, ticks.astype(int))
        ticks = numpy.outer(ticks, ints).reshape(-1)
        return ticks[: sum(ticks <= xMax) + 1]

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class MyCMC:

    _CMCURL = "https://%s-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

    _dataColumns = ["time", "open", "high", "low", "close", "volume"]
    
    _jsonMapper = "./src/BCMapper.json"

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __init__(self, test: bool = False):

        config = ConfigParser()
        config.read("./src/config.ini")
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

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def mapCB(self, cmcs: list[str]):
        """
        CoinMarketCap to Binance shortcut function. Leaves the unfound tickers as they are.\n
        Inputs:\n
            -> "`cmcs`" as list of ticker strs from CoinMarketCap.\n
        Outputs:\n
            -> "`cmcs`" as list of ticker strs from Binance.
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
            -> "`keepTop`" as the amount of top rows being kept in the rank. If None, it is stored as it is.\n
            -> "`saveCSV`" as a bool whether to save the DataFrame as CSV or not, just in case.\n
        Outputs:\n
            -> DataFrame with sorted top market caps (optional, else remains as instance attributes.)
        """
        print("Downloading CMC data from CoinMarketCap. Please wait.")
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
        if saveCSV: self.dataCMC.to_csv("./csv/QuotesCMC.csv") # Store data as CSV if wished.
        self.filterBy = "CMC" # Next time that "getBinance" runs, pre-filter will be done by this result.
        if keepTop: return self.getCMC(keepTop) # If top rows were input, output DataFrame.

    def getCMC(self, keepTop: int = 50) -> Series:
        """
        Fast retrieval of already stored top market cap cryptocurrencies.\n
        Inputs:\n
            -> "`keepTop`" as the amount of top rows desired in the rank.\n
        Outputs:\n
            -> DataFrame holding the top market cap assets.
        """
        return self.dataCMC["m_cap"].iloc[: keepTop]

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def requestDMV(self, keepTop: int = None, saveCSV: bool = False, findMean: str = "onMonth"):
        """
        Request a DataFrame of Binance assets and their info, ranked by Daily Mean Volume.\n
        Inputs:\n
            -> "`keepTop`" as the amount of top rows being kept in the rank. If None, it is stored as it is.\n
            -> "`saveCSV`" as a bool whether to save the DataFrame as CSV or not, just in case.\n
        Outputs:\n
            -> DataFrame with sorted top daily volumes caps (optional, else remains as instance attributes.)
        """
        binance = Binance(self.configBinance) # Create API session.
        args = {"limit": 365, "timeframe": "1d"}
        self.dataDMV = DataFrame()
        print("Downloading DMV data from Binance. Please wait.")
        for entry in binance.fetch_markets():
            if (entry["quote"] != "USD"): continue
            volumes = binance.fetch_ohlcv(entry["symbol"], **args)
            volumes = numpy.array([candle[-1] for candle in volumes])
            self.dataDMV.at[entry["base"], "onWeek"] = volumes[-7 :].mean()
            self.dataDMV.at[entry["base"], "onMonth"] = volumes[-31 :].mean()
            self.dataDMV.at[entry["base"], "onYear"] = volumes[-365 :].mean()
        
        self.dataDMV["main"] = self.dataDMV[findMean]
        if saveCSV: self.dataDMV.to_csv("./csv/QuotesDMV.csv") # Store data as CSV if wished.
        self.filterBy = "DMV" # Next time that "getBinance" runs, pre-filter will be done by this result.
        if keepTop: return self.getDMV(keepTop) # If top rows were input, output DataFrame.

    def getDMV(self, keepTop: int = 50) -> Series:
        """
        Fast retrieval of already stored top daily volume cryptocurrencies.\n
        Inputs:\n
            -> "`keepTop`" as the amount of top rows desired in the rank.\n
            -> "`findMean`" as a string defining if the mean should be calculated `onWeek`-ly, `onMonth`-ly
            or `onYear`-ly basis.\n
        Outputs:\n
            -> DataFrame holding the top market cap assets.
        """
        return self.dataDMV.sort_values("main", ascending = False)["main"].iloc[: keepTop]

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def plotFiltered(self, figsize: tuple[float] = (15, 5), saveJPG: bool = False) -> figure:
        """
        Get a simple histogram showing the market caps of the top assets from larger to smaller.
        Amount of displayed assets depends on the figure width.\n
        Inputs:\n
            -> "`figsize`" as a 2-element tuple: (figure width, figure height).\n
            -> "`saveJPG`" as a bool whether wishing to store the figure as an in-folder JPG file or not.\n
        Outputs:\n
            -> "Figure" handle from Matplotlib.
        """
        # Number of assets/bars proportional to figure width.
        keepTop = figsize[0] * 50 / 15
        keepTop = int(numpy.ceil(keepTop / 10) * 10)
        keepTop = min(self.dataCMC.shape[0], keepTop)
        if (self.filterBy == "CMC"):
            preFiltered = self.getCMC(keepTop = keepTop)
            title = f"Top-%d CoinMarketCap\n" % keepTop
            yLabel = "Market cap (Billion USD)"
        if (self.filterBy == "DMV"):
            preFiltered = self.getDMV(keepTop = keepTop)
            title = f"Top-%d DailyMeanVolume\n" % keepTop
            yLabel = "1000 x traded units"
        # Figure creation and customization.
        axes = preFiltered.plot.bar(figsize = figsize, color = "orange")
        axes.set_yscale("log")
        axes.set_xticks(range(keepTop)) 
        xticks = preFiltered.index.astype(str)
        axes.set_xticklabels(xticks, fontsize = 12, rotation = 90)
        yLogRange = TickRange.log(preFiltered.min(), preFiltered.max())
        # Vertical tick grid generating.
        axes.set_yticks(yLogRange)
        axes.set_yticklabels(yLogRange, fontsize = 12)
        axes.set_ylabel(yLabel, fontsize = 12, fontweight = "bold")
        axes.set_ylim(*yLogRange[[0, -1]]) # Fit grid to logRange.
        title += "‾" * int(len(title) * 1.2) # Title "underlining".
        axes.set_title(title, fontsize = 14, fontweight = "bold", va = "top")
        axes.grid(True, axis = "y", alpha = 1/4, color = "gray", lw = 3)
        axes.figure.set_tight_layout((0, 0, 1, 1)) # Make everything visible.
        if saveJPG: axes.figure.savefig("./fig/Top CMC %s.jpg" % keepTop, dpi = 100)
        return axes.figure

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def getBinance(self, timeFrame: str, since: str = None, keepTop: int = 50, saveCSV: bool = False):
        """
        Get "OHLC candles and volume" market data from the top-ranked market cap assets, and format
        it in a 2-level multi-indexed (asset, OHLCV) DataFrame.\n
        Inputs:\n
            -> "`timeFrame`" as the time "step" of each candle. Examples. "5m", "15m", "1h", "4h", etc.\n
            -> "`since`" as the timestamp (YYYY-MM-DD) of the oldest candle that the DataFrame must keep.\n
            -> "`keepTop`" as the amount of top rows being kept in the market cap rank.\n
            -> "`saveCSV`" as a bool whether it's wished to keep the candle DataFrame as CSV just in case.\n
        Outputs:\n
            -> None. DataFrame is stored as an instance attribute ("[instance]`.dataBinance`").
        """
        binance = Binance(self.configBinance) # Create API session.
        if (since == None): # Make "since" an year ago if None.
            since = datetime("now", utc = True) - Timedelta(365, "day")
            since = since.strftime("%Y-%m-%d")
        args = {"limit": 50000, "timeframe": timeFrame}
        # Check if any CMC ticker is differently named in Binance.
        if (self.filterBy == "CMC"): quotes = self.getCMC(keepTop).index
        if (self.filterBy == "DMV"): quotes = self.getDMV(keepTop).index
        quotes = self.mapCB(quotes)
        # Create an empty dict to hold future candle data.
        self.dataBinance = dict.fromkeys(quotes)
        notFoundQuotes = list()

        progBar = ProgBar(quotes, 40, "Downloading OHLCV data from Binance")
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
            self.dataBinance.reset_index().to_csv("./csv/%s.csv" % title, index = False)

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

if (__name__ == "__main__"):

    myCMC = MyCMC()
    myCMC.requestCMC(keepTop = 50)
    myCMC.plotCMC(saveJPG = True)
    myCMC.getBinance(timeFrame = "4h")
    print(myCMC.dataBinance)