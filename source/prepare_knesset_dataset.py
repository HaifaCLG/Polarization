import json
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import numpy as np
import warnings

from stratified_sampling import stratified_sample
from params_and_config import *
warnings.filterwarnings("ignore")
np.random.seed(42)
STRATIFIED=False
def stratified_sample_df(df, cols, n_samples):
    samples = []
    for col in cols:
        n = min(round(n_samples/cols/df[col].value_counts().min()), df[col].value_counts().min())
        df_ = df.groupby(col).apply(lambda x: x.sample(n))
        df_.index = df_.index.droplevel(0)
        samples.extend(df_.to_dict('records'))
    samples_df = pd.DataFrame.from_records(samples)

    return samples_df
def write_only_sentences_text_to_file_from_csv(all_protocols_chunk_container, output_file_name="all_sentences_text.txt"):
    for chunk in all_protocols_chunk_container:
        texts = list(chunk['sentence_text'].values)
        with open(output_file_name, 'a', encoding='utf-8') as file:
            for text in texts:
                file.write(f'{text}\n')

def write_only_sentences_text_to_file_from_jsonl(jsonl_file, output_file):
    texts = []
    with open(jsonl_file, encoding="utf-8") as file:
        for line in file:
            try:
                json_line = json.loads(line)
                line_text = json_line["sentence_text"]
                texts.append(line_text)
            except Exception as e:
                print(f'couldnt load json. error was:{e}')
                continue
        with open(output_file, 'a', encoding='utf-8') as file:
            for text in texts:
                file.write(f'{text}\n')

def create_sampled_dataset(sampled_csv_output):
    chosen_rows = []
    per_knesset_csvs_path = "D:\\data\\gili\\processed_knesset\\knesset_data_csv_files\\per_knesset_csv_files"
    for file in os.listdir(per_knesset_csvs_path):
        knesset_num = file.split("_")[1]

        file_path = os.path.join(per_knesset_csvs_path, file)
        df = pd.read_csv(file_path, encoding="utf-8")
        df.fillna("None")
        df = df[df["sentence_text"].str.contains("ישיב") == False]
        df = df[df["sentence_text"].str.contains("ועד") == False]
        df = df[df["sentence_text"].str.contains("- -") == False]
        df = df[df["sentence_text"].str.contains("– –") == False]
        df = df[df["sentence_text"].str.contains("--") == False]
        df = df.drop_duplicates('sentence_text')
        df = df[df["sentence_text"].str.split().str.len()>1]


        if STRATIFIED:
            if int(knesset_num) < 16:
                colums_to_stratify_by = ["speaker_id", "turn_num_in_protocol", "protocol_name"]
            else:
                colums_to_stratify_by = ["protocol_type", "speaker_id", "turn_num_in_protocol", "protocol_name"]

            # Get the value counts of the classes
            for col in colums_to_stratify_by:
                class_counts = df[col].value_counts()
                # Get the least populated class
                try:
                    least_populated_values = [x for x in class_counts[class_counts == 1].index]
                except Exception as e:
                    least_populated_class = []
                # Filter the DataFrame to exclude samples from the least populated class
                df = df[~df[col].isin(least_populated_values)]

            stratified = stratified_sample(df=df, strata=colums_to_stratify_by, size=0.01)
            samples = stratified.to_dict('records')
        else:
            sampled_df = df.sample(frac=0.001)
            samples = sampled_df.to_dict('records')
        chosen_rows.extend(samples)
    chosen_rows = random.sample(chosen_rows, 5000)
    df = pd.DataFrame.from_records(chosen_rows)
    df.to_csv(sampled_csv_output)

def write_only_text_from_sentences_shards(input_dir, output_path_dir):
    files = os.listdir(input_dir)
    for file in files:
        file_path = os.path.join(input_dir, file)
        index = file.split("shard_")[1].split(".jsonl")[0]
        if "plenary" in file:
            session_name = "plenary"
        elif "committee" in file:
            session_name = "committee"
        else:
            print(f'wrong file!')
            return
        output_file_name = f'{session_name}_sentences_text_shard_{index}.txt'
        output_file = os.path.join(output_path_dir, output_file_name)
        write_only_sentences_text_to_file_from_jsonl(file_path, output_file)


if __name__ == '__main__':
    # sampled_data_csv_path = os.path.join(knesset_data_csv_files, "knesset_sampled_sentences.csv")
    # create_sampled_dataset(sampled_data_csv_path)
    # chunk_container = pd.read_csv(sampled_data_csv_path, chunksize=1000)
    # write_only_sentences_text_to_file_from_csv(chunk_container, "sampled_knesset_sentences_text.txt")
    write_only_text_from_sentences_shards(committee_full_sentences_shards_path, os.path.join(knesset_txt_files_path,"committee_text_shards"))
    write_only_text_from_sentences_shards(plenary_full_sentences_shards_path, os.path.join(knesset_txt_files_path, "plenary_text_shards"))