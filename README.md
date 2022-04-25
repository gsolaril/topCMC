Just jump directly to <a href="https://github.com/gsolaril/topCMC/blob/main/Main%20report.ipynb">the Jupyter Notebook report</a>.
<br>Feel free to revise the rest of the code files, though.

<hr>

**Changes after Apr/23/2022**:
- Simplified the alpha-ranking procedure. Removed the scoring based on word repetition and counting, and applied a more straight-forward scoring method using the Pandas' "`.rank`" function. As the former leans on a triangular summation and the latter is a mergesort sort, the complexity is theoretically reduced from "$O(n^2)$" to "$O(n log(n))$".
- The resulting `DataFrame` from the procedure is now a `bool`ean grid-like "checklist" instead of a list of `str`ings/quotes. The main motive for this, is that each column/quote, that now holds `True` when it is top-ranked, can work as a "trade/no-trade" signal time-series.
- Elaborated a vectorized backtesting scheme for a long-alpha strategy based on the principle in the point above. Can backtest just one quote, or a portfolio of them.
- Returns, drawdown and statistical metrics are now calculated on the results of such vectorized backtesting scheme.

**Changes after Apr/17/2022**:
- Removed the selection process that dealt with long-term and short-term average log-return histograms. If needed, it shall be still present in past commits anyway.
- In its place, added an alpha-ranking scheme with a scoring method to filter the different pre-selected (50-100) instruments according to their alpha values, and their recurrence among high alpha values for each day. Read motivation and description a bit before half of the notebook.
- Added another preselection filter, based in top daily average volumes (averaged over last week, last month and last year). Function is called "`requestDMV`" and uses Binance API.
- Cleaned a bit along the coding cells: many of them were appended to ".py" files, and described with comments.
- Reorganized the repo, with subfolders and sections.

Thanks for reading :)
