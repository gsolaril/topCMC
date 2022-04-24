import numpy
from pandas import Series, Timedelta, DataFrame, MultiIndex
from matplotlib.pyplot import figure, subplots
from matplotlib.figure import Figure, Axes
from pandas import to_datetime as datetime
from IPython.display import display as print
from itertools import product
from .ProgBar import *

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Appendix:
    @staticmethod
    def f_abs(x: Series): return numpy.abs(x)
    @staticmethod
    def f_log(x: Series): return numpy.log(x)
    @staticmethod
    def f_sign(x: Series): return numpy.sign(x)
    @staticmethod
    def f_rank(x: Series): return x.rank(pct = True) #.iloc[-1]
    @staticmethod 
    def f_delay(x: Series, d: int = 0): return x.shift(d).bfill()
    @staticmethod
    def f_scale(x: Series, a: int = 1): return a * (x / x.sum())
    @staticmethod
    def f_delta(x: Series, d: int = 1): return x.diff(d) #.iloc[-1]
    @staticmethod
    def f_signedpower(x: Series, a: float): return numpy.power(x, a)
    @staticmethod
    def f_correlation(x: Series, y: Series, d = 1): return x.rolling(d).corr(y)
    @staticmethod
    def f_covariance(x: Series, y: Series, d = 1): return x.rolling(d).cov(y)
    @staticmethod
    def f_decay_linear(x: Series, d: int):
        weights = numpy.arange(d) + 1
        wma = lambda x: numpy.dot(x, weights)
        return x.rolling(d).apply(wma) / weights.sum()
    @staticmethod
    def f_mean(x: Series, d: int): return x.rolling(d).mean()
    @staticmethod
    def f_ts_min(x: Series, d: int): return x.rolling(d).min()
    @staticmethod
    def f_ts_max(x: Series, d: int): return x.rolling(d).max()
    @staticmethod
    def f_ts_argmin(x: Series, d: int): return x.rolling(d).apply(numpy.argmin)
    @staticmethod
    def f_ts_argmax(x: Series, d: int): return x.rolling(d).apply(numpy.argmax)
    @staticmethod
    def f_ts_rank(x: Series, d: int):
        return x.rolling(d).apply(lambda x:
            x.size - x.argsort().argsort()[-1]) / d
    @staticmethod
    def f_sum(x: Series, d: int): return x.rolling(d).sum()
    @staticmethod
    def f_product(x: Series, d: int): return x.rolling(d).apply(numpy.prod)
    @staticmethod
    def f_stddev(x: Series, d: int): return x.rolling(d).std()

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

class Alpha:

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    @staticmethod # chosen 1
    def alpha_1(close: Series, **kwargs) -> Series:

        returns = close.pct_change()
        cond_2 = returns.apply(lambda x: x < 0)
        true_2, false_2 = Appendix.f_stddev(returns, 20), close
        arg_2 = (cond_2 * true_2) + (~cond_2 * false_2)
        arg_1 = Appendix.f_signedpower(arg_2, 2)
        arg_0 = Appendix.f_ts_argmax(arg_1, 5)
        return Appendix.f_rank(arg_0) - 0.5

    @staticmethod # chosen 2
    def alpha_10(close: Series, **kwargs) -> Series:

        arg_1 = Appendix.f_delta(close, 1)
        cond_1 = Appendix.f_ts_min(arg_1, 4).apply(lambda x: x > 0)
        true_1 = Appendix.f_delta(close, 1)
        arg_2 = Appendix.f_delta(close, 1)
        cond_2 = Appendix.f_ts_max(arg_2, 4).apply(lambda x: x < 0)
        true_2 = Appendix.f_delta(close, 1)
        false_2 = -Appendix.f_delta(close, 1)
        false_1 = (cond_2 * true_2) + (~cond_2 * false_2)
        arg_0 = (cond_1 * true_1) + (~cond_1 * false_1)
        return Appendix.f_rank(arg_0)

    @staticmethod # chosen 3
    def alpha_11(close: Series, volume: Series, **kwargs) -> Series:

        vwap = (close * volume).cumsum() / volume.cumsum()
        arg_4 = vwap - close
        arg_3a = Appendix.f_ts_max(arg_4, 3)
        arg_3b = Appendix.f_ts_min(arg_4, 3)
        fact_2a = Appendix.f_rank(arg_3a) + Appendix.f_rank(arg_3b)
        fact_2b = Appendix.f_rank(Appendix.f_delta(volume, 3))
        return fact_2a * fact_2b
    
    @staticmethod # chosen 4
    def alpha_20(open: Series, high: Series, low: Series, close: Series, **kwargs) -> Series:

        fact_0 = Appendix.f_rank(open - Appendix.f_delay(high, 1))
        fact_1 = Appendix.f_rank(open - Appendix.f_delay(close, 1))
        fact_2 = Appendix.f_rank(open - Appendix.f_delay(low, 1))
        return - fact_0 * fact_1 * fact_2

    @staticmethod  # chosen 5
    def alpha_21(close: Series, volume: Series, **kwargs) -> Series: ## CHECK

        adv20 = Appendix.f_mean(volume, 20)
        term_3 = Appendix.f_sum(close, 8) / 8 + Appendix.f_stddev(close, 8)
        term_2a = Appendix.f_sum(close, 2) / 2
        term_2b = Appendix.f_sum(close, 8) / 8 - Appendix.f_stddev(close, 8)
        cond_2 = (term_2a - term_2b).apply(lambda x: x < 0)
        cond_3 = (volume / adv20).apply(lambda x: 1 <= x)
        true_3, false_3 = 1, -1
        false_2 = (cond_3 * true_3) + (~cond_3 * false_3)
        true_1, true_2 = -1, 1
        false_1 = (cond_2 * true_2) + (~cond_2 * false_2)
        cond_1 = (term_3 - Appendix.f_sum(close, 2) / 2).apply(lambda x: x < 0)
        return (cond_1 * true_1) + (~cond_1 * false_1)

    @staticmethod # chosen 6
    def alpha_30(close: Series, volume: Series, **kwargs) -> Series:

        term_1 = Appendix.f_sign(close - Appendix.f_delay(close, 1))
        term_2 = Appendix.f_sign(Appendix.f_delay(close, 1) - Appendix.f_delay(close, 2))
        term_3 = Appendix.f_sign(Appendix.f_delay(close, 2) - Appendix.f_delay(close, 3))
        term_0 = 1.0 - Appendix.f_rank(term_1 + term_2 + term_3)
        fact_0 = Appendix.f_sum(volume, 5) / Appendix.f_sum(volume, 20)
        return term_0 * fact_0

    @staticmethod # chosen 7
    def alpha_31(low: Series, close: Series, volume: Series, **kwargs) -> Series:

        adv20 = Appendix.f_mean(volume, 20)
        arg_2a = -Appendix.f_rank(Appendix.f_rank(Appendix.f_delta(close, 10)))
        arg_1a = Appendix.f_decay_linear(arg_2a, 10)
        term_1a = Appendix.f_rank(Appendix.f_rank(arg_1a))
        term_1b = Appendix.f_rank(-Appendix.f_delta(close, 3))
        arg_1c = Appendix.f_correlation(adv20, low, 12)
        term_1c = Appendix.f_sign(Appendix.f_scale(arg_1c))
        return term_1a + term_1b + term_1c

    @staticmethod # chosen 8
    def alpha_40(high: Series, volume: Series, **kwargs) -> Series:

        fact_0 = - Appendix.f_rank(Appendix.f_stddev(high, 10))
        fact_1 = Appendix.f_correlation(high, volume, 10)
        return fact_0 * fact_1

    @staticmethod # chosen 9
    def alpha_60(high: Series, low: Series, close: Series, volume: Series, **kwargs) -> Series:

        num_3 = ((close - low) - (high - close)) * volume
        div_3 = (high - low)
        arg_3 = num_3 / div_3
        term_1a = 2 * Appendix.f_scale(Appendix.f_rank(arg_3))
        arg_2 = Appendix.f_ts_argmax(close, 10)
        term_1b = Appendix.f_scale(Appendix.f_rank(arg_2))
        term_0 = term_1a - term_1b
        return -term_0

    @staticmethod # chosen 10
    def alpha_101(open: Series, high: Series, low: Series, close: Series, **kwargs) -> Series:

        return (close - open) / (0.001 + high - low)
    
    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    @classmethod
    def available(cls) -> dict:

        alphas = [x for x in dir(cls) if ("alpha_" in x)]
        nAlpha = [int(x.replace("alpha_", "")) for x in alphas]
        fAlpha = [getattr(cls, x) for x in alphas]
        return dict(zip(nAlpha, fAlpha))

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __init__(self, values: DataFrame, cPrices: DataFrame):

        self.values, self.cPrices = values, cPrices

    @classmethod
    def fromOHLCV(cls, OHLCV: DataFrame):
        """
        Calculates given Alpha functions for the whole "`OHLCV`" `DataFrame` holding horizontally
        stacked market data timeseries. Gives an "`Alpha`" instance, where "`.values`" holds the
        numerical results.\n
        Inputs:\n
            -> "`OHLCV`": Multi-indexed `DataFrame`. Upper column level ("`OHLCV.columns.levels[0]`")
            should have instrument labels (quotes), and lower level ("`OHLCV.columns.levels[1]`") must
            hold characteristic OHLCV columns ("`["open", "high", "low", "close", "volume]`").\n
        Outputs:\n
            -> `None`
        """
        alphas = Alpha.available() # Dict with all (labelled) alpha calculators. See previous method.
        quotes, columns = OHLCV.columns.levels
        alphaColumns = MultiIndex.from_product((alphas.keys(), quotes))
        # "values" will have the same structure as OHLCV, but with alpha ints instead of OHLCV level.
        values = DataFrame(index = OHLCV.index, columns = alphaColumns) 
        OHLCV = OHLCV.swaplevel(axis = "columns") # iterate over OHLCV, not over quotes.
        try: OHLCV[columns]
        except: OHLCV = OHLCV.swaplevel(axis = "columns")
        cPrices = OHLCV["close"] # Keep close prices for future graphical comparisons (".alphaplot")

        quoteVerbose = list()
        for n, quote in enumerate(quotes, 1):
            verbose = "%s (%d/%d)" % (quote, n, len(quotes))
            quoteVerbose.append(verbose)

        items = product(alphas.keys(), quoteVerbose)
        items = ["%d for %s" % (alpha, quote) for alpha, quote in items]
        progBar = ProgBar(items = items, width = 40, verbose = "Alpha")
        for nAlpha, funcAlpha in alphas.items():
            for quote in quotes:
                values.loc[:, (nAlpha, quote)] = funcAlpha(**{
                    column: OHLCV[column][quote] for column in columns})
                progBar.show()
        
        return cls(values, cPrices)

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def rank(self, nRank: int, chosenAlphas: list[int] = [1, 101, 10, 40, 30], plot: bool = False):
        """
        This method will build a "`.ranked`" quote" DataFrame (instance attribute) which will
        take the form of a grid. "`.index`" is the OHLCV timeline, while "`.columns`" are the
        different tested quotes.\n "Filled" cells ("`True`") for a certain row (datetime) represent
        those quotes that display large enough values of the "`chosenAlphas`" to be ranked as
        top-trending quotes, feasible for trading.\n The max amount of cells per row that are
        allowed to be filled/"`True`" is given by "`nRank`".\n
        Inputs:\n
            -> "`nRank`", the max amount of quotes are "`.ranked`" to be `True` in the final ranking.\n
            -> "`chosenAlphas`", a list of which particular alphas should the method contemplate.
        Outputs:\n
            -> A figure with a Gantt-like representation of the "`.ranked`"\n
        NOTE: Generated `scores` are linear; proportional to the "`.rank`" in each single ranking.
        A good idea however, would be a scoring scheme based on alpha values itself.
        But the orders of magnitude seem to differ so much from one alpha to another.
        So this shall be left for future developments.
        """
        timeline = self.values.index
        quotes = self.values.columns.levels[1]
        alphaReduced = self.values[chosenAlphas]
        # ".ranked" DataFrame sets "True" to the "nKeep" quotes...
        # ...with overall highest alpha values for each candle timestamp.
        self.ranked = DataFrame(False, columns = quotes, index = timeline)
        
        items = ["row %d (%s)" % (r, t.strftime("%Y-%m-%d")) for r, t in enumerate(timeline)]

        progBar = ProgBar(items = items, width = 40, verbose = "Ranking")
        for time in timeline:
            eachAlphaRanking = alphaReduced.loc[time].unstack()
            eachAlphaRanking = eachAlphaRanking.rank(axis = "columns").fillna(0)
            eachAlphaRanking = eachAlphaRanking.astype(int).sum(axis = "index")
            eachAlphaRanking = eachAlphaRanking.sort_values(ascending = False)
            eachAlphaRanking = eachAlphaRanking.index[: nRank]
            self.ranked.loc[time, eachAlphaRanking] = True
            progBar.show()

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def qualified(self):
        """
        Columnar (not grid) representation of "`.ranked`". Returns a list-like `DataFrame` with the
        labels of the top-ranked quotes that were found during the execution of "`.rank`". Useful for
        easy display of previously filtered quote sets.\n
        Inputs:\n
            -> `None`\n
        Outputs:\n
            -> `None`\n
        """
        nRank = self.ranked.sum(axis = "columns")
        nRank = numpy.arange(nRank.max())[:: -1] + 1
        timeline = self.ranked.index
        quotes = self.ranked.columns
        qualified = DataFrame(index = timeline, columns = nRank)
        for time in timeline:
            row = self.ranked.loc[time]
            qualified.loc[time] = quotes[row]
        return qualified.iloc[:, :: -1]

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def plotAlpha(self, last: int = 100, quoteLeft: str = None,
            quoteRight: str = None, chosenAlphas: list[int] = None) -> Figure:
        """
        Stack multiple alpha plots vertically in a single compact figure. Will compare two tickers'
        alpha series at a time ("left" and "right"). That shall be enough to draw comparisons and
        observations with respect to price series. The idea is to avoid overlapping and messy plots.\n
        Inputs:\n
            -> "`last`" is the amount of most recent timeseries' rows to be plotted.\n
            -> "`quoteLeft`" and "`quoteRight`" are the tickers to be plotted. If not specified or
            `None`, they will be chosen randomly.\n
            -> "`chosenAlphas`" is the list of alphas to plot. If not specified, all are considered. 
        Outputs:\n
            -> "figure" with all stacked plots: the one on the top represents price, and the others
            below are the alphas. Alpha numbers lie on the rightmost border.
        """
        alphas, quotes = self.values.columns.levels
        if (chosenAlphas == None): chosenAlphas = list(alphas)
        if (quoteLeft == None): quoteLeft = numpy.random.choice(quotes) 
        if (quoteRight == None): quoteRight = numpy.random.choice(quotes)
        assert (quoteLeft in quotes), "Please choose to plot an existent quote."
        assert (quoteRight in quotes), "Please choose to plot an existent quote."
        alphaReduced = self.values.swaplevel(axis = "columns")
        alphaReduced = alphaReduced[[quoteLeft, quoteRight]].iloc[-last :]
        items = chosenAlphas
        nRows = len(items) + 1
        figure, axes = subplots(figsize = (15, nRows * 1.5), nrows = nRows, sharex = True)

        alphAxes = dict(zip(items, axes[1 :]))
        xTicks = alphaReduced.index[:: alphaReduced.shape[0] // 50]

        color_L, color_R = ["lime", "tomato"] # change colors here.
        ax_L, ax_R = axes[0], axes[0].twinx()
        ax_L.grid(True, lw = 3, alpha = 1/3)
        price_L, price_R = self.cPrices[[quoteLeft, quoteRight]].iloc[-last :].values.T
        ax_L.plot(alphaReduced.index, price_L, color = color_L, lw = 3)
        ax_L.set_ylabel(quoteLeft, fontsize = 16, fontweight = "bold", color = color_L)
        ax_L.tick_params(labelsize = 12, labelcolor = color_L)
        ax_L.set_ylim(price_L.min(), price_L.max()) ; ax_L.minorticks_off()
        ax_R.plot(alphaReduced.index, price_R, color = color_R, lw = 3)
        ax_R.set_ylabel(quoteRight, fontsize = 16, fontweight = "bold", color = color_R)
        ax_R.tick_params(labelsize = 12, labelcolor = color_R)
        ax_R.set_ylim(price_R.min(), price_R.max()) ; ax_R.minorticks_off()
        title = "Chosen alpha values for %s and %s" % (quoteLeft, quoteRight)
        title += "\n" + "‾" * int(len(title) * 1.2)
        ax_L.set_title(title, fontsize = 14, fontweight = "bold")
        
        progBar = ProgBar(items + ["Generating figure"], 40, "Drawing alpha")
        for nAlpha, alphAxis in alphAxes.items():

            alphAxis.plot(alphaReduced.index, alphaReduced[quoteLeft][nAlpha],
                label = quoteLeft, axes = alphAxis, color = color_L, lw = 2)
            alphAxis.plot(alphaReduced.index, alphaReduced[quoteRight][nAlpha],
                label = quoteRight, axes = alphAxis, color = color_R, lw = 1.5)
            alphAxis.grid(True, lw = 3, alpha = 1/3)
            alphAxis.set_ylabel("\nAlpha #%s" % nAlpha,
                fontweight = "bold", fontsize = 12)
            alphAxis.tick_params(labelsize = 12)
            alphAxis.yaxis.set_label_position("right")
            alphAxis.set_xticks([]) ; progBar.show()
            alphAxis.minorticks_off()

        alphAxis.set_xticks(xTicks)
        alphAxis.set_xticklabels(xTicks.strftime("%m/%d %H:%M"), rotation = 90)
        alphAxis.set_xlim(xTicks[0], xTicks[-1]) ; progBar.show()
        figure.set_tight_layout((0, 0, 1, 1))
        return figure

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def plotRanked(self) -> Figure:
        """
        Returns a "Gantt-like" grid that describes the incidence and frequency of the ".qualified" quotes
        that resulted from the "`.rank`" method along time. Those that are are more regularly top-ranked,
        will draw on upper regions. \n
        Inputs:\n
            -> `None`. Just run "`.rank`" before this...
        Outputs:\n
            -> Figure handle
        """
        qualifiedSort = self.ranked.sum(axis = "index").sort_values()
        qualifiedGrid = self.ranked.loc[:, qualifiedSort.index]
        qualifiedGrid = qualifiedGrid.replace(False, numpy.nan)
        timeline = qualifiedGrid.index
        quotes = qualifiedGrid.columns
        qualifiedGrid = (qualifiedGrid - 0.5) + range(len(quotes))
        fig, ax = subplots(figsize = (15, len(quotes) // 5))

        title = "Recurrence of cryptocurrencies in top-ranked alphas from"
        title += " %s to %s" % (*timeline.strftime("%Y-%m-%d")[[0, -1]],)
        ax.set_title(title)

        color = ["tomato", "limegreen", "skyblue", "magenta", "pink"]
        color = numpy.tile(color, len(quotes) // len(color) + 1)[: len(quotes)]
        qualifiedGrid.plot(ax = ax, marker = "|", ms = 10, lw = 0, color = color)
        ax.get_legend().remove()

        xTicks = timeline[: : len(timeline) // 50]
        ax.set_xticks(xTicks)
        ax.set_xlim(xTicks[0], xTicks[-1])
        xTicks = xTicks.strftime("%Y-%m-%d")
        ax.set_xticklabels(xTicks, fontsize = 12, rotation = 90);
        
        ax.set_yticks(numpy.arange(len(quotes)) + 0.5)
        ax.set_yticklabels(quotes, fontsize = 14)
        ax.set_xlabel(None)
        
        ax.set_ylim(0, len(quotes))
        ax.grid(True, lw = 1, alpha = 1, ls = "--")
        return fig

#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

if (__name__ == "__main__"): # Tests

    dt = Timedelta(value = 1, unit = "days")
    t = datetime("2022-01-01") + numpy.arange(500) * dt
    p = Series(numpy.random.randn(t.size), index = t)
    p = 1234 + (1.2 * p).cumsum().round(2)
    DF = p.resample(rule = "5D").agg("ohlc")
    columns = ["open", "high", "low", "close"]
    alphArgs = {k: DF[k] for k in columns}
    c = alphArgs["close"]
    v = 10 * (c + numpy.random.randn(c.size) * 2)
    alphArgs["volume"] = v

    d = numpy.random.randint(1, 10)
    print("d-parameter: %d\n" % d)

    #methods = list(globals().items())[-20 :]
    for method in dir(Appendix):
        if ("__" in method): continue
        if not ("f_" in method): continue
        function = getattr(Appendix, method)
        nArgs = function.__code__.co_argcount
        sep = "-" * 30 + " " + method + ":\n...\n%s\n"
        if (nArgs == 1): print(sep % function(c).tail(d + 2)) ;  continue
        if (nArgs == 2): print(sep % function(c, d).tail(d + 2)) ; continue
        if (nArgs == 3): print(sep % function(c, v, d).tail(d + 2)) ; continue

    print("ALPHAS:\n------\n")
    for method in dir(Alpha):
        if not ("alpha_") in method: continue
        function = getattr(Alpha, method)
        nArgs = function.__code__.co_argcount
        sep = "-" * 30 + " " + method + ":\n...\n%s"
        print(sep % function(**alphArgs).tail(d + 2))
        