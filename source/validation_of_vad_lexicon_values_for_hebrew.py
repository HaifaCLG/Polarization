import sys

import pandas as pd
import numpy as np
import random
from scipy.stats import kendalltau, pearsonr, spearmanr


random.seed(42)

def calculate_mean_pairwise_annotators_agreement(scores_path_list):
    scores_dfs = []
    for path, i in zip(scores_path_list, range(len(scores_path_list))):
        scores_dfs.append(load_scores(path).rename(columns={'score': f'score{i+1}'}))
    # Merge on the 'word' column to align words
    df = scores_dfs[0].merge(scores_dfs[1], on='word').merge(scores_dfs[2], on='word')
    # Rank the scores for each annotator
    df['rank1'] = df['score1'].rank(method='average', ascending=True)
    df['rank2'] = df['score2'].rank(method='average', ascending=True)
    df['rank3'] = df['score3'].rank(method='average', ascending=True)

    # Compute pairwise Kendall's tau
    tau_12, _ = kendalltau(df['rank1'], df['rank2'])
    tau_13, _ = kendalltau(df['rank1'], df['rank3'])
    tau_23, _ = kendalltau(df['rank2'], df['rank3'])

    # Mean pairwise Kendall's tau
    mean_tau = (tau_12 + tau_13 + tau_23) / 3


    print(f"Pairwise Kendall's tau:")
    print(f"Annotator 1 & 2: {tau_12:.4f}")
    print(f"Annotator 1 & 3: {tau_13:.4f}")
    print(f"Annotator 2 & 3: {tau_23:.4f}")
    print(f"Mean pairwise Kendall's tau: {mean_tau:.4f}")

    #pearson:
    pearson_corr_12, _ = pearsonr(df['score1'], df['score2'])
    pearson_corr_13, _ = pearsonr(df['score1'], df['score3'])
    pearson_corr_23, _ = pearsonr(df['score2'], df['score3'])
    mean_pearson_corr = (pearson_corr_12+pearson_corr_13+pearson_corr_23)/3
    print(f"Mean pairwise pearson corr: {mean_pearson_corr:.4f}")


def compare_annotators_to_gold(scores_path_list, gold_scores_path, vad_column_name):
    # Load annotators' scores
    scores_dfs = []
    for path, i in zip(scores_path_list, range(len(scores_path_list))):
        scores_dfs.append(load_scores(path).rename(columns={'score': f'score{i + 1}'}))

    # Merge on the 'word' column to align words
    df = scores_dfs[0]
    for df_other in scores_dfs[1:]:
        df = df.merge(df_other, on='word')

    # Compute average of annotators' scores
    df['mean_score'] = df[[f'score{i + 1}' for i in range(len(scores_path_list))]].mean(axis=1)

    # Rescale mean_score from [-1, 1] to [0, 1] to match gold range
    df['mean_score_scaled'] = (df['mean_score'] + 1) / 2

    # Load gold scores and merge by word
    df_gold = pd.read_csv(gold_scores_path).rename(columns={vad_column_name: 'gold_score', 'Hebrew Word': 'word'})
    df = df.merge(df_gold, on='word')

    # Compute Pearson's correlation (on rescaled mean scores vs. gold scores)
    pearson_corr, _ = pearsonr(df['mean_score_scaled'], df['gold_score'])

    # Compute Kendall's tau (on rescaled mean scores vs. gold scores)
    kendall_corr, _ = kendalltau(df['mean_score_scaled'], df['gold_score'])
    rho, _ = spearmanr(df['mean_score_scaled'], df['gold_score'])


    print(f"Comparison to Gold Standard:")
    print(f"Spearman’s rho: {rho:.4f}")
    print(f"Pearson correlation (rescaled mean vs. gold): {pearson_corr:.4f}")
    print(f"Kendall's tau (rescaled mean vs. gold): {kendall_corr:.4f}")


def load_scores(file_path):
    # Each line: "word <tab> score"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['word', 'score'])
    return df

# Rank the scores for Kendall's tau
def rank_scores(df):
    # Rank the scores in order (highest score gets highest rank)
    df['rank'] = df['score'].rank(method='average', ascending=True)
    return df

# Compute pairwise Kendall's tau
def compute_pairwise_kendall(df1, df2):
    tau, _ = kendalltau(df1['rank'], df2['rank'])
    return tau

def select_words_for_manual_annotation(csv_path: str, VAD_column_name: str, sample_size=100) -> pd.DataFrame:
    """
    Loads a CSV, sorts it by the specified column, then for each consecutive 100-word block
    in the 'English Word' column, selects one English word at random whose corresponding
    'Hebrew Undotted' value is unique across the entire DataFrame. Returns a DataFrame
    of selected English words, their Hebrew Undotted forms, and the sort column values.
    """
    # Load and sort
    df = pd.read_csv(csv_path, na_filter=True)
    df = df.replace({np.nan: None, "": None})
    df_sorted = df.sort_values(by=VAD_column_name).reset_index(drop=True)

    results = []

    chunk_size = int(np.ceil(len(df_sorted) / sample_size))
    num_chunks = int(np.ceil(len(df_sorted)/chunk_size))
    for chunk_idx in range(num_chunks):
        start, end = chunk_idx * chunk_size, (chunk_idx + 1) * chunk_size
        chunk = df_sorted.iloc[start:end]
        candidates = chunk.index.tolist()
        selected_idx = None

        # Sample until a unique Hebrew Undotted is found or candidates exhausted
        while candidates:
            idx = random.choice(candidates)
            heb = df_sorted.at[idx, 'Hebrew Undotted']
            if not heb:
                heb = df_sorted.at[idx, 'Hebrew Word']
            heb = heb.strip()
            # Count occurrences of heb across the entire DataFrame
            occurrences = (df_sorted.applymap(lambda x: str(x) == heb).values.sum())
            if occurrences == 1:
                selected_idx = idx
                break
            candidates.remove(idx)

        if selected_idx is None:
            raise ValueError(f"No unique Hebrew Undotted found in chunk {chunk_idx} (rows {start}-{end}).")

        row = df_sorted.loc[selected_idx]
        if row['Hebrew Undotted']:
            hebrew_word = row['Hebrew Undotted']
        else:
            hebrew_word = row['Hebrew Word']
        results.append({
            'English Word': row['English Word'],
            'Hebrew Word': hebrew_word,
            f'{VAD_column_name}_value': row[VAD_column_name]
        })
    return pd.DataFrame(results)


def prepare_words_for_tuple_script(words_for_hebrew_annotation_csv, file_name):
    df = pd.read_csv(words_for_hebrew_annotation_csv,  encoding='utf-8-sig')
    hebrew_words = df['Hebrew Word'].dropna()
    hebrew_words.to_csv(
        file_name,
        index=False,
        header=False,
        encoding='utf-8'
    )

def make_excel_file_for_annotators(file_path, output_excel):
    try:
        # Read the TSV (no header)
        df = pd.read_csv(file_path, sep='\t', header=None, dtype=str)
    except Exception as e:
        print(f"Error reading '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if df.shape[1] != 4:
        print(
            f"Warning: expected 4 columns in input but found {df.shape[1]}. "
            "Proceeding anyway and naming the first four A–D.",
            file=sys.stderr
        )

        # 3) Rename first four columns to A, B, C, D
    cols = ["Item1", "Item2", "Item3", "Item4" ]
    for i, name in enumerate(cols):
        if i < df.shape[1]:
            df.rename(columns={i: name}, inplace=True)
        else:
            # if missing columns, create them as empty
            df[name] = ""

    # 4) Add the two empty columns
    df["BestItem"] = ""
    df["WorstItem"] = ""

    # 5) Reorder to ensure A–D, Highest, Lowest
    df = df[cols + ["BestItem", "WorstItem"]]

    if output_excel:
        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")

                worksheet = writer.sheets['Sheet1']
                # freeze the top row (so row 1 stays visible when you scroll)
                worksheet.freeze_panes = 'A2'
        except Exception as e:
            print(f"Error writing Excel to '{output_excel}': {e}", file=sys.stderr)
            sys.exit(1)

    if output_excel:
        print(f"Successfully wrote Excel to {output_excel}")

def mix_words_in_file(words_file):
    with open(words_file, encoding="utf-8-sig") as file:
        words = file.readlines()
    random.shuffle(words)
    mixed_file_name = f'{words_file.replace(".txt","_shuffled.txt")}'
    with open(mixed_file_name, 'w', encoding='utf-8') as f:
        for item in words:
            f.write(f"{item.strip()}\n")

def check_how_many_translations_have_more_than_two_words(lexicon_path):
    df = pd.read_csv(lexicon_path, na_filter=True)
    df = df.replace({np.nan: None, "": None})
    counter = 0
    for index, row in df.iterrows():
        hebrew_word = row['Hebrew Undotted']
        if not hebrew_word:
            hebrew_word = row['Hebrew Word']
        if hebrew_word:
            num_of_words_in_term = len(hebrew_word.split())
        else:
            print(f'error: empty word')
            continue
        if num_of_words_in_term>2:
            english_word = row["English Word"]
            counter+=1
            print(f'{english_word}: {hebrew_word}')
    print(f'number of hebrew translations with over 2 words: {counter}')
    num_rows = len(df)
    print("Number of English terms:", num_rows)


if __name__ == '__main__':
    lexicon_path = enriched_final_lexicon_path
    check_how_many_translations_have_more_than_two_words(lexicon_path)
    vad_hebrew_words_annotation_path = os.path.join(data_path, "vad_manual_annotations_hebrew", "vad_hebrew_words_annotation")
    valence_words_for_hebrew_annotation_csv_file = os.path.join(vad_hebrew_words_annotation_path,"final_valence_words_for_hebrew_annotation.csv")
    valence_selected_df = select_words_for_manual_annotation(lexicon_path, 'Valence')
    valence_selected_df.to_csv(valence_words_for_hebrew_annotation_csv_file,index=False, encoding='utf-8-sig')
    arousal_words_for_hebrew_annotation_csv_file = os.path.join(vad_hebrew_words_annotation_path,"final_arousal_words_for_hebrew_annotation.csv")
    arousal_selected_df = select_words_for_manual_annotation(lexicon_path, 'Arousal')
    arousal_selected_df.to_csv(arousal_words_for_hebrew_annotation_csv_file,index=False, encoding='utf-8-sig')

    dominance_words_for_hebrew_annotation_csv_file = os.path.join(vad_hebrew_words_annotation_path,"final_dominance_words_for_hebrew_annotation.csv")
    dominance_selected_df = select_words_for_manual_annotation(lexicon_path, 'Dominance')
    dominance_selected_df.to_csv( dominance_words_for_hebrew_annotation_csv_file,index=False, encoding='utf-8-sig')
    valence_words_input_for_tuple_script = "final_valence_words_input_for_tuple_script.txt"
    arousal_words_input_for_tuple_script = "final_arousal_words_input_for_tuple_script.txt"
    dominance_words_input_for_tuple_script = "final_dominance_words_input_for_tuple_script.txt"
    prepare_words_for_tuple_script(valence_words_for_hebrew_annotation_csv_file, valence_words_input_for_tuple_script)
    prepare_words_for_tuple_script(arousal_words_for_hebrew_annotation_csv_file, arousal_words_input_for_tuple_script)
    prepare_words_for_tuple_script(dominance_words_for_hebrew_annotation_csv_file, dominance_words_input_for_tuple_script)


    make_excel_file_for_annotators( os.path.join(vad_hebrew_words_annotation_path,"final_valence_words_input_for_tuple_script.txt.tuples"), 'valence_words_for_annotators.xlsx')
    make_excel_file_for_annotators(os.path.join(vad_hebrew_words_annotation_path,"final_arousal_words_input_for_tuple_script.txt.tuples"), 'arousal_words_for_annotators.xlsx')
    make_excel_file_for_annotators(os.path.join(vad_hebrew_words_annotation_path,"final_dominance_words_input_for_tuple_script.txt.tuples"), 'dominance_words_for_annotators.xlsx')

    gili_v_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","gili_v_scores.txt")
    ella_v_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","ella_v_scores.txt")
    shuly_v_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","shuly_v_scores.txt")
    gili_a_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","gili_a_scores.txt")
    ella_a_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","ella_a_scores.txt")
    shuly_a_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","shuly_a_scores.txt")
    gili_d_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","gili_d_scores.txt")
    ella_d_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","ella_d_scores.txt")
    shuly_d_scores_path = os.path.join(vad_hebrew_words_annotation_path,"annotators_scores","shuly_d_scores.txt")

    annotators_v_scores_list = [gili_v_scores_path,ella_v_scores_path,shuly_v_scores_path]
    annotators_a_scores_list = [gili_a_scores_path,ella_a_scores_path,shuly_a_scores_path]
    annotators_d_scores_list = [gili_d_scores_path,ella_d_scores_path,shuly_d_scores_path]
    calculate_mean_pairwise_annotators_agreement(annotators_v_scores_list)
    calculate_mean_pairwise_annotators_agreement(annotators_a_scores_list)
    calculate_mean_pairwise_annotators_agreement(annotators_d_scores_list)
    v_gold_path = valence_words_for_hebrew_annotation_csv_file
    vad_column_name = "Valence_value"
    compare_annotators_to_gold(annotators_v_scores_list, v_gold_path, vad_column_name)

    a_gold_path = arousal_words_for_hebrew_annotation_csv_file
    vad_column_name = "Arousal_value"
    compare_annotators_to_gold(annotators_a_scores_list, a_gold_path, vad_column_name)

    d_gold_path = dominance_words_for_hebrew_annotation_csv_file
    vad_column_name = "Dominance_value"
    compare_annotators_to_gold(annotators_d_scores_list, d_gold_path, vad_column_name)
