%matplotlib inline
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta

e = pd.DataFrame()
e.ta.indicators()
help(ta.sma)
df = df.rename({'Close': 'close', 'Volume':'volume'}, axis=1)  # new method
macddf = df.ta.bbands(length = 5)
macddf

