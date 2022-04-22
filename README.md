Just jump directly to <a href="https://github.com/gsolaril/topCMC/blob/main/Main%20report.ipynb">the Jupyter Notebook report</a>.
<br>Feel free to revise the rest of the code files, though.

<hr>

Changes after Apr/17/2022:
- Removed the selection process that dealt with long-term and short-term average log-return histograms. If needed, it shall be still present in past commits anyway.
- In its place, added an alpha-ranking scheme with a scoring method to filter the different pre-selected (50-100) instruments according to their alpha values, and their recurrence among high alpha values for each day. Read motivation and description a bit before half of the notebook.
- Added another preselection filter, based in top daily average volumes (averaged over last week, last month and last year). Function is called "`requestDMV`" and uses Binance API.
- Cleaned a bit along the coding cells: many of them were appended to ".py" files, and described with comments.
- Reorganized the repo, with subfolders and sections.

Thanks for reading :)
