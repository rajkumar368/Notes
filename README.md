# Pandas All-in-One Notes  
```python
# =========================================================
# 0. IMPORT
# =========================================================
import pandas as pd
import numpy as np
from io import StringIO

# =========================================================
# 1. CREATING DATA
# =========================================================
# from dict
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# from list of dicts
df = pd.DataFrame([{'A': 1, 'B': 4}, {'A': 2, 'B': 5}])

# from numpy
df = pd.DataFrame(np.random.randn(5, 3), columns=list('ABC'))

# Series
s = pd.Series([1, 3, 5, np.nan])

# MultiIndex
midx = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['letter', 'number'])
midx_df = pd.DataFrame(np.random.randn(4, 2), index=midx, columns=['X', 'Y'])

# DatetimeIndex
dates = pd.date_range('2025-01-01', periods=6, freq='D')
ts = pd.Series(np.random.randn(6), index=dates)

# =========================================================
# 2. I/O  (CSV, JSON, EXCEL, SQL, PARQUET, HDF5, CLIPBOARD)
# =========================================================
df.to_csv('out.csv', index=False)
df = pd.read_csv('out.csv')

df.to_json('out.json', orient='records')
df = pd.read_json('out.json')

df.to_excel('out.xlsx', index=False, sheet_name='Sheet1')
df = pd.read_excel('out.xlsx', sheet_name='Sheet1')

# SQL (needs SQLAlchemy)
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///test.db')
# df.to_sql('table', engine, index=False)
# df = pd.read_sql('SELECT * FROM table', engine)

df.to_parquet('out.parquet')
df = pd.read_parquet('out.parquet')

df.to_hdf('store.h5', key='df')
df = pd.read_hdf('store.h5', key='df')

# clipboard
df.to_clipboard(index=False)
df = pd.read_clipboard()

# =========================================================
# 3. INSPECTION
# =========================================================
df.head()
df.tail()
df.info(memory_usage='deep')
df.describe(include='all')
df.shape
df.columns
df.dtypes
df.memory_usage(deep=True)
df.select_dtypes(include=['number'])

# =========================================================
# 4. INDEXING & SLICING
# =========================================================
df['A']               # one column
df[['A', 'C']]        # many columns
df[0:3]               # slice rows
df.loc[0]             # by label
df.loc[0:2, 'A':'C']  # slice label
df.iloc[0]            # by position
df.iloc[0:2, 0:2]     # slice position
df.at[0, 'A']         # scalar
df.iat[0, 0]          # scalar by position
df.query('A > 0 & B < 0')
df[df['A'].isin([1, 2])]

# =========================================================
# 5. ASSIGN / MODIFY / DROP
# =========================================================
df['new'] = df['A'] + df['B']
df.insert(1, 'Z', [99, 99, 99])
df.assign(C_times_D=lambda d: d['A'] * d['B'])
df.drop(columns=['new'])
df.drop_duplicates()
df.dropna()
df['A'] = df['A'].astype('int32')

# =========================================================
# 6. MISSING DATA
# =========================================================
df.isna()
df.notna()
df.fillna(0)
df.fillna(method='ffill')
df.fillna(method='bfill')
df.interpolate()
df.dropna(axis=0)  # drop rows
df.dropna(axis=1)  # drop cols

# =========================================================
# 7. SORTING
# =========================================================
df.sort_values('A')
df.sort_values(['A', 'B'], ascending=[True, False])
df.sort_index()

# =========================================================
# 8. FILTERING & WHERE
# =========================================================
df[df['A'] > 0]
df.where(df['A'] > 0, other=0)
df.mask(df['A'] > 0, other=0)

# =========================================================
# 9. AGGREGATIONS
# =========================================================
df.sum()
df.mean()
df.median()
df.std()
df.var()
df.min()
df.max()
df.quantile([0.25, 0.75])
df.nunique()
df.count()
df.cumsum()
df.cumprod()
df.cummin()
df.cummax()

# =========================================================
# 10. GROUPBY
# =========================================================
g = df.groupby('A')
g.size()
g.sum()
g.agg(['sum', 'mean', 'std'])
g.agg({'B': 'sum', 'C': 'mean'})
g.transform(lambda x: x - x.mean())
g.filter(lambda x: x['B'].mean() > 0)

# =========================================================
# 11. MERGE, JOIN, CONCAT, APPEND
# =========================================================
df1 = pd.DataFrame({'key': ['A', 'B'], 'val1': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'val2': [3, 4]})
pd.merge(df1, df2, on='key')
pd.concat([df1, df2], axis=0)
df1.append(df2, ignore_index=True)
df1.join(df2.set_index('key'), on='key')

# =========================================================
# 12. PIVOT, MELT, CROSSTAB
# =========================================================
pivot = df.pivot_table(values='C', index='A', columns='B', aggfunc='mean', fill_value=0)
melt = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
ct = pd.crosstab(df['A'], df['B'])

# =========================================================
# 13. STRING OPERATIONS
# =========================================================
s = pd.Series(['  hello ', 'world'])
s.str.strip()
s.str.upper()
s.str.lower()
s.str.replace('hello', 'hi')
s.str.extract(r'(\w+)')
s.str.contains('hello')
s.str.split()

# =========================================================
# 14. DATETIME
# =========================================================
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()
df.set_index('date', inplace=True)
df.resample('M').mean()
df.rolling(window=7).mean()

# =========================================================
# 15. CATEGORICAL
# =========================================================
df['cat'] = df['cat'].astype('category')
df['cat'] = df['cat'].cat.set_categories(['low', 'mid', 'high'], ordered=True)

# =========================================================
# 16. PERFORMANCE
# =========================================================
df = df.copy()
df.eval('D = A + B')
df.query('A > 0')
df['D'] = df['A'] + df['B']  # vectorized

# =========================================================
# 17. APPLY / MAP / APPLYMAP
# =========================================================
df['A'].apply(lambda x: x**2)
df[['A', 'B']].applymap(lambda x: x**2)
df['A'].map({1: 'one', 2: 'two'})

# =========================================================
# 18. WINDOW FUNCTIONS
# =========================================================
df['roll_mean'] = df['A'].rolling(window=3).mean()
df['exp_mean'] = df['A'].ewm(span=3).mean()
df['shift'] = df['A'].shift(1)
df['diff'] = df['A'].diff()

# =========================================================
# 19. RANK & QUANTILE
# =========================================================
df['rank'] = df['A'].rank(method='dense')
df['qcut'] = pd.qcut(df['A'], q=4)

# =========================================================
# 20. VISUALIZATION (quick)
# =========================================================
df.plot()
df.plot(kind='bar')
df.plot(kind='hist')
df.plot(kind='box')
df.plot.scatter(x='A', y='B')

# =========================================================
# 21. EXPORT MULTIPLE SHEETS
# =========================================================
with pd.ExcelWriter('report.xlsx', engine='openpyxl') as w:
    df.to_excel(w, sheet_name='raw', index=False)
    pivot.to_excel(w, sheet_name='pivot')

# =========================================================
# 22. ADVANCED EXAMPLES
# =========================================================
# Example: full pipeline
raw = pd.read_csv('sales.csv')
clean = (raw
         .drop_duplicates()
         .dropna(subset=['price'])
         .assign(
             revenue=lambda d: d['price'] * d['qty'],
             month=lambda d: pd.to_datetime(d['date']).dt.to_period('M')
         )
         .groupby(['month', 'region'])
         .agg(total=('revenue', 'sum'))
         .reset_index()
         .pivot(index='month', columns='region', values='total')
         .fillna(0)
)
clean.to_excel('monthly_report.xlsx')

# =========================================================
# 23. ONE-LINER CHEATS
# =========================================================
# clipboard → quick view
pd.read_clipboard().head()
# quick summary
df.describe(include='all').T
# memory usage
df.info(memory_usage='deep')
# shape
print(df.shape)
# sample
df.sample(5)
# unique counts
df.nunique()
# correlations
df.corr()
# null counts
df.isna().sum()
# duplicate counts
df.duplicated().sum()
