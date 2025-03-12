import pandas as pd
import taipy.gui as tpg

# mettiamo solo il test set!
df = (
    pd.read_parquet("data/out/df.parquet")
    .reset_index()
    .rename(columns={"index": "timestamp"})
)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

page = """
# 📈 Web App con Taipy

### 📋 Dataset
<|{df.head(10_000)}|table|width=100%|>

### 📊 Grafico Interattivo
<|{df.head(10_000)}|chart|x=timestamp|y=s4_mean|type=line|>
"""

if __name__ == "__main__":
    tpg.Gui(page).run(title="Scintill-AI", dark_mode=True)
