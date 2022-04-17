import numpy
from pandas import Series, Timedelta
from pandas import to_datetime as datetime
from IPython.display import display as print

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

    @classmethod
    def available(cls) -> dict:

        alphas = [x for x in dir(cls) if ("alpha_" in x)]
        nAlpha = [int(x.replace("alpha_", "")) for x in alphas]
        fAlpha = [getattr(cls, x) for x in alphas]
        return dict(zip(nAlpha, fAlpha))

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
        