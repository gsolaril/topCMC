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

    def rank(self, nKeep: int, nRank: int = 5, chosenAlphas: list[int] = [1, 101, 10, 40, 30]):
        """
        This method will build a "`GlobalRanking`" DataFrame (called "`.ranking`" as instance attribute)
        whose objective is to keep the everyday "nKeep" quotes/tickers which appear the most frequently
        when single rankings are done for each one of the calculated alphas. \nQuotes with the highest
        rank inside each alpha's single ranking should have priority over the others, so a simple linear
        scoring system is implemented. Then, the final "`nKeep`" quotes over each day are considered, and
        the others are disregarded. \nReading code comments is advised to clear any further doubts.
        Inputs:\n
            -> "`nKeep`", as how many quotes are pretended to be kept in each row of the final ranking.\n
            -> "`nRank`", as how many quotes should consider while first ranking by each single alpha.\n
            -> "`chosenAlphas`", as a list of which particular alphas should the method contemplate.
        Outputs:\n
            -> `None`. ("`GlobalRanking`" is generated and assigned to "`.ranking`" as a result).\n
        NOTE: Generated `scores` are linear; proportional to the rank in each single ranking.
        A good idea however, would be a scoring scheme based on alpha values itself.
        But the orders of magnitude seem to differ so much from one alpha to another.
        So this shall be left for future developments.
        """
        nKept = range(1, nKeep + 1)
        nRanked = range(1, nRank + 1)
        alphaReduced = self.values[chosenAlphas]
        alphaReduced = alphaReduced.resample("1D").agg("mean")
        # "SingleRanking" will hold the "nRank" quotes with the highest ranked values...
        alphaSingleRanking = DataFrame(index = alphaReduced.index, # ...for each date...
            columns = MultiIndex.from_product((chosenAlphas, nRanked))) # ...for each Single alpha.
        # "GlobalRanking" will keep the "nKeep" quotes that appear the most in each SingleRanking row...
        alphaGlobalRanking = DataFrame(index = alphaReduced.index, columns = nKept) # ...for each date.

        timeline = alphaSingleRanking.index              # scores = -> 1st (highest) place: "nRank",
        items = product(timeline.strftime("%Y-%m-%d"), chosenAlphas) # 2nd place: "nRank - 1", ...
        scores = nRank - numpy.tile(range(nRank), len(chosenAlphas)) # Last (lowest) place: "1".
        items = ["row at %s, alpha #%d" % (time, alpha) for time, alpha in items]

        progBar = ProgBar(items = list(items), width = 40, verbose = "Ranking")
        for time in timeline:

            skipRow = False
            for nAlpha in chosenAlphas:
                progBar.show()
                # jump over row without valid values, given the lack of info to compare any further.
                if alphaReduced.loc[time, nAlpha].isna().all(): skipRow = True ; continue
                else: # get an amount of "nFilter" quote names that hold the largest value of each alpha.
                    alphaSorted = alphaReduced.loc[time, nAlpha].sort_values()
                    alphaSingleRanking.loc[time, nAlpha] = alphaSorted.index[: nRank]
                    
            if skipRow: continue                  # e.g.: for "chosen =  [ ↓ 1            , ↓ 2 ]"
            ranked = alphaSingleRanking.loc[time] + " " # and "nRank = 3": [BTC, ETH, LTC], [XMR, ETH, DOG]
            ranked = scores * ranked # [BTC, BTC, BTC, ETH, ETH, LTC], [XMR, XMR, XMR, ETH, ETH, DOG]
            repeats = " ".join(ranked).split(" ") # [BTC, BTC, BTC, ETH, ETH, LTC, XMR, XMR, XMR, ETH...
            repeats = Series(filter(len, repeats)) # Erases empty strings. Solved a bug.
            repeats = repeats.value_counts() # {"ETH": 4, "XMR": 3, "BTC": 3, "LTC": 1, "DOGE": 1}
            newQuotes = repeats.index[: nKeep] # for "nKeep = 2": [ETH, XMR]
            toFill = range(1, len(newQuotes) + 1) # in case less than "nKeep" elements were found.
            alphaGlobalRanking.loc[time, toFill] = newQuotes
        
        self.ranking = alphaGlobalRanking.copy() # Drop any incomplete row.

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def alphaplot(self, last: int = 100, quoteLeft: str = None,
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

    def alphaplot2(self, color: str = "red") -> Figure:
        """
        Returns a "Gantt-like" chart that describes the incidence and frequency of the instruments in
        the Global Ranking. Instruments which are more regularly top-ranked, appear on upper regions.
        It's important that "`.rank`" method was executed first.\n
        Inputs:\n
            -> "`color`" of the frequency markers.
        Outputs:\n
            -> Figure handle
        """
        alphaRankingFreq = Series(self.ranking.values.reshape(-1))
        alphaRankingFreq = alphaRankingFreq.value_counts().index
        height = round(len(alphaRankingFreq) / 5, 1)
        fig, ax = subplots(figsize = (15, height))
        ax.set_yticks(numpy.arange(len(alphaRankingFreq)) + 0.5)
        ax.set_yticklabels(alphaRankingFreq, fontsize = 14);
        timeline = self.ranking.index
        xTicks = timeline[: : len(timeline) // 50]
        ax.set_xticks(xTicks)
        ax.set_xlim(xTicks[0], xTicks[-1])
        xTicks = xTicks.strftime("%Y-%m-%d")
        ax.set_xticklabels(xTicks, fontsize = 12, rotation = 90);
        title = "Recurrence of cryptocurrencies in top-%d alpha ranking from"
        title += " %s to %s" % (*timeline.strftime("%Y-%m-%d")[[0, -1]],)
        ax.set_title(title % self.ranking.shape[1])

        alphaRankingHeights = self.ranking.replace(dict(zip(
            alphaRankingFreq[::-1], range(alphaRankingFreq.size))))

        rankArgs = {"marker": "s", "ms": 3, "color": color, "lw": 0}
        for rank in alphaRankingHeights.columns:
            line = alphaRankingHeights[rank] + 0.5
            ax.plot(line.index, line.values, **rankArgs)
        ax.set_ylim(0, len(alphaRankingFreq))
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
        