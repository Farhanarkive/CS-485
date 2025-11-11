import pandas as pd

df = pd.read_csv("train.csv")

data = df.sample(n=10000)

data.to_csv("sample.csv")

