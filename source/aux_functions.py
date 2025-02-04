import codecs
import os.path
import pickle
import random
import shutil

import pandas as pd
from conllu import parse_incr, parse
from itertools import islice
from langdetect import detect
import re


def create_small_sample_of_df(df, num_of_entries, save_as_csv=False, new_df_file_name="sampled_df.csv"):
    new_df = df.sample(n=num_of_entries)
    if save_as_csv:
        new_df.to_csv(new_df_file_name)
    return new_df



def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

def is_hebrew(sentence, print_non_hebrew=True):
    try:
        # Detect language
        lang = detect(sentence)
        # Check if the language is Hebrew
        if lang == 'he':
            return True
        else:
            if print_non_hebrew:
                print(f'non hebrew sentence: {sentence}')
            return False
    except:
        # In case of any detection error, consider the sentence not Hebrew
        if print_non_hebrew:
            print(f'non hebrew or defected sentence: {sentence}')
        return False

def save_object(obj, object_name):
    obj_name = object_name.replace("/", "_")
    path = os.path.join('pickle_objects', f'{obj_name}.pkl')
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object(object_name):
    obj_name = object_name.replace("/", "_")
    path = os.path.join('pickle_objects', f'{obj_name}.pkl')
    with open(path, 'rb') as file:
        model_obj = pickle.load(file)
    return model_obj


def parse_each_conllu_sentence_separatly(data):
    sentences = []
    separated_data = data.split("\n\n")
    for conllu_sent in separated_data:
        try:
            sent = parse(conllu_sent)
        except:
            continue
        sentences.extend(sent)
    return sentences


def merge_two_large_csv_files(first_csv_file, second_csv_file, output_file_name):
    CHUNK_SIZE = 50000
    chunk_container = pd.read_csv(first_csv_file, chunksize=CHUNK_SIZE)
    First_time = True
    for chunk in chunk_container:
        if First_time:
            chunk.to_csv(output_file_name, mode="a", index=False)
            First_time = False
        else:
            chunk.to_csv(output_file_name, mode="a", index=False, header=False)

    chunk_container = pd.read_csv(second_csv_file, chunksize=CHUNK_SIZE, skiprows=[0])
    for chunk in chunk_container:
        chunk.to_csv(output_file_name, mode="a", index=False, header=False)


def sample_n_files_from_dir(dir_path, n, folder_to_copy_files=None):
    file_names = os.listdir(dir_path)
    chosen_file_names = random.sample(file_names, n)
    if folder_to_copy_files:
        for file in chosen_file_names:
            original_path = os.path.join(dir_path, file)
            dest_path = os.path.join(folder_to_copy_files, file)
            shutil.copy(original_path, dest_path)
    return chosen_file_names
