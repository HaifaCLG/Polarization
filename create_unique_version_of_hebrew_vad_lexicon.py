import pandas as pd

def is_english(word: str) -> int:
    """
    Returns 1 if 'word' is purely ASCII (English),
    otherwise 0. We'll use this for sorting so that
    Hebrew (non-ASCII) words come first.
    """
    try:
        # If this encoding fails, the word has non-ASCII chars
        word.encode('ascii')
        return 1  # purely English
    except UnicodeEncodeError:
        return 0  # has Hebrew/non-ASCII

#  Read the CSV.
#    Adjust "words.csv" to your actual filename/path.
df = pd.read_csv("path\to\vad_full_hebrew_enriched_lexicon.csv", index_col=0)

#  Remove empty or NaN Hebrew Words.
df = df[df["Hebrew Word"].notnull()]                # remove NaN
df = df[df["Hebrew Word"].str.strip() != ""]        # remove empty strings

#  Group by "Hebrew Word" and average numeric columns (Valence, Arousal, Dominance).
df_grouped = df.groupby("Hebrew Word", as_index=False).mean()

#  Separate and sort:
#    - Create a temporary column (is_eng) to mark English words with 1, Hebrew words with 0.
#    - Sort by that flag ascending (so Hebrew first, English second), then by word alphabetically.
df_grouped["is_eng"] = df_grouped["Hebrew Word"].apply(is_english)
df_sorted = df_grouped.sort_values(by=["is_eng", "Hebrew Word"], ascending=[True, True])

#  Drop the helper column "is_eng" before saving.
df_sorted = df_sorted.drop(columns=["is_eng"])

# Save to a new CSV without the index column.
df_sorted.to_csv("path\to\\unique_vad_full_hebrew_enriched_lexicon.csv", index=False)

print("New CSV file 'words_averaged_sorted.csv' has been created.")
