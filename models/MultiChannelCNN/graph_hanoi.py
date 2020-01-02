import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("fake_results.csv")
agent_zero = df[df["agent"] == 0]