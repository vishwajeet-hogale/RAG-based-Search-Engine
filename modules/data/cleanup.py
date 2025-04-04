import pandas as pd

if __name__ == "__main__":
    # df = pd.read_csv("professor_info.csv")
    # df = df[["Research Area","Professor Name","Publication Date","Publication Title","Publication Link","Citation","Publication Summary"]]
    # df.to_csv("final_research_info.csv")

    # df1 = pd.read_csv("labs_with_summaries.csv")
    # df1 = df1[["Research Area","Lab Name","Summary"]]
    # df1.to_csv("final_lab_summaries.csv")

    df1 = pd.read_csv("professor_details.csv")
    df1 = df1[["Research Area","Professor Name","Biography","Research Interests","Education"]]
    df1 = df1.drop_duplicates()
    # Group by Professor Name and aggregate Research Area
    combined_df = df1.groupby("Professor Name", as_index=False).agg({
        "Research Area": lambda x: "; ".join(sorted(set(x.dropna()))),  # merge unique, non-null areas
        "Biography": "first",             # keep the first non-null biography
        "Research Interests": "first",    # or you can use .agg(lambda x: x.dropna().iloc[0]) for robustness
        "Education": "first"
    })

    # Save to final CSV
    combined_df.to_csv("final_prof_details.csv", index=False)
    # df1.to_csv("final_prof_details.csv")