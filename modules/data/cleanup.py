import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("professor_info.csv")
    df = df[["Research Area","Professor Name","Publication Date","Publication Title","Publication Link","Citation","Publication Summary"]]
    df.to_csv("professor_info.csv")