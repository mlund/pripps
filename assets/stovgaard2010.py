# See https://doi.org/10.1186/1471-2105-11-429

# Convert CSV file to Pripps YAML form factor definition file

import pandas as pd

df = pd.read_csv("stovgaard2010.csv")

for residue in df.columns.to_list():
    if residue == "q" or residue == "Unnamed: 0":
        continue
    print(f"{residue}: !Table [", end="")
    for q, I in zip(df["q"], df[residue]):
        print(f"[{q}, {I}], ", end="")
    print("]")

