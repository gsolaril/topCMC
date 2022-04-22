import os, sys, numpy
from scipy import stats
from matplotlib.pyplot import figure, subplot, style
from pandas import Series, DataFrame, Timestamp, Timedelta
from MyCMC import *
style.use("https://raw.githubusercontent.com/gsolaril/"\
 + "Templates4Coding/master/Python/mplfinance.mplstyle")

class Portfolio:

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __init__(self, cap: float,
            assetLabels: list[str],      # tickers
            assetLongVols: list[float]): # long-term volatilities

        self.cap = cap ; self.labels = assetLabels
        meanVol = numpy.mean(assetLongVols) # establish mean volatility as benchmark
        self.sizes = meanVol / assetLongVols # invest inversely prop to volatility
        self.sizes = self.sizes / sum(self.sizes) # should normalize to percentages
        self.sizes = self.cap * self.sizes # "sizes" should now include cash amount
        self.btDF = self.ftDF = self.stDF = None # placeholders for future results

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def backtest(self, logReturns: DataFrame, plot: bool = True):

        self.btDF = DataFrame(columns = self.labels, index = logReturns.index)

        self.logReturns = logReturns
        cumReturns = numpy.exp(logReturns.cumsum()) # "cumReturns" keeps time series of capital growth in %.
        self.btDF = self.sizes * cumReturns # so mult. by each cash amount, should keep capital growth in $. 
        self.btDF["total"] = self.btDF.sum(axis = "columns") # recalculate returns.
        self.btDF["logReturns"] = self.btDF["total"].pct_change().fillna(0)
        self.btDF["logReturns"] = numpy.log(1 + self.btDF["logReturns"])
        cumLogReturns = self.btDF["logReturns"].cumsum() + 1
        self.btDF["cumReturns"] = numpy.exp(cumLogReturns - 1) - 1
        self.btDF["drawdown"] = 1 - cumLogReturns / cumLogReturns.cummax()
        #if plot: return self.__plotBacktest()

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#
    
    def __plotBacktest(self):

        toPlot = self.btDF[list(self.labels) + ["total"]].copy()
        fig, axes = subplot(nrows = 2, figsize = (15, 10))

        axes = toPlot.plot(ax = axes)
        axes.set_yscale("log")
        yLogRange = toPlot.min().min(), toPlot.max().max()
        yLogRange = MyCMC.logRange(*yLogRange)
        axes.set_yticks(yLogRange)
        yTicks = [str(x / 1e6) + "M" for x in yLogRange]
        axes.set_yticklabels(yTicks, fontsize = 12)

        title = "Portfolio backtest, %s - %s" % (t1, t2)
        title += "\n" + "‾" * int(len(title) * 1.2)
        axes.set_title(title, fontsize = 14, fontweight = "bold")
        axes.tick_params("x", labelsize = 12, rotation = 90)

        xTicks = toPlot.index
        xTicks = xTicks[:: len(xTicks) // 50]
        axes.set_xlim(xTicks[0], xTicks[-1])
        axes.grid(True, lw = 3, alpha = 0.5)
        axes.set_xticks(xTicks) ; axes.set_xlabel("")
        axes.legend(fontsize = 14, ncol = len(lowCorr) + 1)
        axes.figure.set_tight_layout((0, 0, 1, 1))
        axes.figure.savefig("./csv/%s.jpg" % title.split("\n")[0], dpi = 120)
        return fig

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def forwardtest(self, until: Timestamp, sims: int = 100, plot: bool = True):

        assert isinstance(self.btDF, DataFrame), \
            "\".backtest\" should be executed first."
        delta = self.btDF.index[-1] - self.btDF.index[-2]
        futureIndex = numpy.arange(self.btDF.index[-1], until, delta)[1 :]
        mean = self.logReturns.mean(axis = "index")
        stdv = self.logReturns.std(axis = "index")
        rMinWalk = rMaxWalk = None # placeholder for best and worst cases.
        progBar = ProgBar(items = sims, width = 40, verbose = "Sim")
        for sim in range(sims):  # sequence of random numbers "N(0, 1)":
            rWalk = numpy.random.randn(len(futureIndex), len(self.labels)) 
            rWalk = DataFrame(rWalk, index = futureIndex, columns = self.labels).shift(1)
            rWalk = 1 + (mean + rWalk * stdv).cumsum() # random numbers "N(mean, stdv)"
            rWalk.fillna(1, inplace = True) # Beginning accounts for starting capital.
            if isinstance(rMinWalk, DataFrame):
                for quote in self.labels:
                    # worst case as the timeseries' area's "floor"
                    rMinWalk[quote] = concat((rMinWalk, rWalk),
                        axis = "columns").min(axis = "columns")
                    # best case as the timeseries' area's "ceiling"
                    rMaxWalk[quote] = concat((rMaxWalk, rWalk),
                        axis = "columns").max(axis = "columns")
            else: rMinWalk, rMaxWalk = rWalk.copy(), rWalk.copy()
            progBar.show()

        endBT = self.btDF[self.labels].iloc[-1]
        # starting cap is past backtest's ending cap:
        rMinWalk *= endBT ; rMaxWalk *= endBT
        rMidWalk = (rMinWalk + rMaxWalk) / 2
        self.ftDF = {"min": rMinWalk, "mid": rMidWalk, "max": rMaxWalk}
        for key, rWalk in self.ftDF.items():
            # re-calc all return timeseries for each case:
            rWalk["total"] = rWalk.sum(axis = "columns")
            rWalk["logReturns"] = rWalk["total"].pct_change().fillna(0)
            rWalk["logReturns"] = numpy.log(1 + rWalk["logReturns"])
            cumLogReturns = rWalk["logReturns"].cumsum() + 1
            rWalk["cumReturns"] = numpy.exp(cumLogReturns - 1) - 1
            rWalk["drawdown"] = 1 - cumLogReturns / cumLogReturns.cummax()
            self.ftDF[key] = rWalk
        
        #if plot: return self.__plotForwardtest()

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def __plotForwardtest(self):

        toPlot = concat(self.ftDF, axis = "columns")
        toPlot = toPlot.swaplevel(axis = "columns")["total"]
        toPlot = toPlot.replace(numpy.inf, numpy.nan)
        axes = toPlot["mid"].plot(figsize = (15, 5), label = "Expected")
        axes.fill_between(alpha = 1/3, label = "Possible area",
            x = toPlot.index, y1 = toPlot["min"], y2 = toPlot["max"])

        yRange = axes.get_yticks()
        yTicks = [str(x / 1e6) + "M" for x in yRange]
        axes.set_yticklabels(yTicks, fontsize = 12)

        t2b = toPlot.index[-1].strftime("%Y-%m-%d")
        title = "Forward projections for "
        title += "portfolio net value, until %s" % t2b
        title += "\n" + "‾" * int(len(title) * 1.2)
        axes.set_title(title, fontsize = 14, fontweight = "bold")
        axes.tick_params("x", labelsize = 12, rotation = 90)

        xTicks = toPlot.index
        xTicks = xTicks[:: len(xTicks) // 50]
        axes.set_xlim(xTicks[0], xTicks[-1])
        axes.grid(True, lw = 3, alpha = 0.5)
        axes.set_xticks(xTicks) ; axes.set_xlabel("")
        axes.set_xticklabels(xTicks.strftime("%Y-%m-%d"))
        axes.legend(fontsize = 14, ncol = len(self.labels) + 1)
        axes.figure.set_tight_layout((0, 0, 1, 1))
        axes.figure.savefig("./csv/%s.jpg" % title.split("\n")[0], dpi = 120)
        return axes.figure

    #===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#===#

    def stats(self):

        assert isinstance(self.btDF, DataFrame), \
            "\".backtest\" should be executed first."
        self.stDF = Series(dtype = float)
        logReturns = self.btDF["total"].pct_change()
        logReturns = numpy.log(1 + logReturns)
        self.stDF["upCandles"] = (logReturns > 0).sum()
        self.stDF["dnCandles"] = (logReturns < 0).sum()
        self.stDF["retMean"] = logReturns.mean()
        self.stDF["retStDv"] = logReturns.std()
        self.stDF["retRange"] = logReturns.max() - logReturns.min()
        self.stDF["Sharpe"] = self.stDF["retMean"] / self.stDF["retStDv"]
        self.stDF["pNLoss"] = stats.norm.cdf(- self.stDF["Sharpe"])
        logReturns_D = logReturns.resample("1D").agg("sum")
        self.stDF["retMean_D"] = logReturns_D.mean()
        self.stDF["retStDv_D"] = logReturns_D.std()
        self.stDF["Sharpe_D"] = self.stDF["retMean_D"] / self.stDF["retStDv_D"]
        self.stDF["pNLoss_D"] = stats.norm.cdf(- self.stDF["Sharpe_D"])
        self.stDF["Sharpe_M"] = numpy.sqrt(30) * self.stDF["Sharpe_D"]
        self.stDF["pNLoss_M"] = stats.norm.cdf(- self.stDF["Sharpe_M"])
        self.stDF["Sharpe_Y"] = numpy.sqrt(365) * self.stDF["Sharpe_D"]
        self.stDF["pNLoss_M"] = stats.norm.cdf(- self.stDF["Sharpe_Y"])
        cumReturns = 1 + logReturns.cumsum()
        drawdown = 1 - cumReturns / cumReturns.cummax()
        self.stDF["avgDrawdown"] = drawdown.mean()
        self.stDF["maxDrawdown"] = drawdown.max()
        self.stDF["Calmar"] = self.stDF["retMean"] / self.stDF["avgDrawdown"]
        self.stDF["Sterling"] = self.stDF["retMean"] / self.stDF["maxDrawdown"]