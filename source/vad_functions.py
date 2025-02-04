import os.path
import random
import re

import pandas as pd
import scipy
import torch
from scipy.stats import pearsonr
from enum import Enum
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import cos_sim

import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from aux_functions import *
from aux_functions import *
from params_and_config import emobank_csv_path, data_path

random.seed(42)
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

HEBREW_ENGLISH_VAD_LEXICON_PATH = os.path.join(data_path,"vad","NRC-VAD-Lexicon","NRC-VAD-Lexicon","OneFilePerLanguage","Hebrew-NRC-VAD-Lexicon.txt")
HEBREW_ENGLISH_EMOTIONS_LEXICON_PATH = os.path.join(data_path,"vad","Hebrew-NRC-EmoLex.txt")

class Language(Enum):
    ENGLISH = "english"
    HEBREW = "hebrew"

LANGUAGE = Language.HEBREW
if LANGUAGE== Language.HEBREW:
    # SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large'
    # SENT_TRANSFORMER_TYPE = 'imvladikon/sentence-transformers-alephbert'
    # SENT_TRANSFORMER_TYPE = "imvladikon/alephbertgimmel-base-512"
    SENT_TRANSFORMER_TYPE = 'dicta-il/dictabert'
    # SENT_TRANSFORMER_TYPE = "yam-peleg/Hebrew-Mistral-7B"
else:
    # SENT_TRANSFORMER_TYPE = 'bert-base-nli-mean-tokens'
    SENT_TRANSFORMER_TYPE = 'bert-large-nli-mean-tokens'
    # SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large'
if SENT_TRANSFORMER_TYPE == 'intfloat/multilingual-e5-large':
    MULTI = False#todo restore this for original multi
else:
    MULTI = False

ALEPHBERTGIMMEL = False
DICTABERT = True
YAM_PELEG = False
OUR_DICTA_PRETRAINED_MODEL = False
OUR_MULTI_PRETRAINED_MODEL = True# change to false for original multi

if OUR_DICTA_PRETRAINED_MODEL:
    SENT_TRANSFORMER_TYPE = 'dicta-il/dictabert-ourfinetuned'

    # SENT_TRANSFORMER_TYPE = 'all-MiniLM-L6-v2'

if OUR_MULTI_PRETRAINED_MODEL:
    SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large-ourfinetuned'

CONVERT_VAD_LEX_TO_CSV = False#should happen only once
RECREATE_ENCODING_MODEL_AND_EMBEDDINGS = True# change back to false for original multi
RECREATE_AND_TRAIN_BINOM_REGRESSION_MODELS = True# change back to false for original multi
USE_ALL_DATA_FOR_TRAINING = True
TRAIN_WITH_KNESSET_DATA = True
TRAIN_WITH_EMO_BANK_DATA = True

if USE_ALL_DATA_FOR_TRAINING:
    Valence_regression_model_name = f'v_binom_model_final_{LANGUAGE.value}_trained_with_knesset_and_emobank_data_{SENT_TRANSFORMER_TYPE}'
    Arousal_regression_model_name = f'a_binom_model_final_{LANGUAGE.value}_trained_with_knesset_and_emobank_data_{SENT_TRANSFORMER_TYPE}'
    Dominance_regression_model_name = f'd_binom_model_final_{LANGUAGE.value}_trained_with_knesset_and_emobank_data_{SENT_TRANSFORMER_TYPE}'
else:
    Valence_regression_model_name = f'v_binom_model_{LANGUAGE.value}_trained_with_{SENT_TRANSFORMER_TYPE}'
    Arousal_regression_model_name = f'a_binom_model_{LANGUAGE.value}_trained_with_{SENT_TRANSFORMER_TYPE}'
    Dominance_regression_model_name = f'd_binom_model_{LANGUAGE.value}_trained_with_{SENT_TRANSFORMER_TYPE}'
def convert_lexicon_file_to_csv(lexicon_path, output_path):
    with open(lexicon_path, encoding='utf-8') as lex_txt_file:
        data = []
        first_line = True
        for line in lex_txt_file:
            line_words = line.split("\t")
            line_words_stripped = [s.strip() for s in line_words]

            if first_line:
                columns = line_words_stripped
                first_line = False
            else:
                data.append(line_words_stripped)
        df = pd.DataFrame.from_records(data, columns=columns)
        df.to_csv(output_path)

def select_knesset_sentences(fraction=1):
    knesset_poll_sentences_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\vad-sentence-annotations\\Best-Worst-Scaling-Scripts\\scored\\knesset_poll_sentences_avg_scores_all_vad.csv"
    df = pd.read_csv(knesset_poll_sentences_path, index_col=None)
    df = df.sample(frac=fraction, random_state=42)
    print(f'knesset total sentences: {len(df)}')
    sentences = df["sent_text"].to_list()

    return sentences, df


def select_n_unique_items_k_times_knesset_sentences(n_train=96, k=5):
    output_path = os.path.join('pickle_objects', f'{"slice_df_and_sentences_list"}.pkl')
    if os.path.exists(output_path):
        slice_df_and_sentences_list = load_object("slice_df_and_sentences_list")
        return slice_df_and_sentences_list
    knesset_poll_sentences_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\vad-sentence-annotations\\Best-Worst-Scaling-Scripts\\scored\\knesset_poll_sentences_avg_scores_all_vad.csv"
    df = pd.read_csv(knesset_poll_sentences_path, index_col=None)
    total = len(df)
    n_test = total - n_train

    if total < n_test * k:
        raise ValueError(f"Not enough samples to create {k} unique test sets of {n_test} items each.")

    # Shuffle the DataFrame once
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    slice_df_and_sentences_list = []

    for i in range(k):
        start_idx = i * n_test
        end_idx = start_idx + n_test

        df_test = df_shuffled.iloc[start_idx:end_idx]
        df_train = df_shuffled.drop(df_test.index).sample(n=n_train, random_state=42)

        sentences_train = df_train["sent_text"].to_list()

        slice_df_and_sentences_list.append((sentences_train, df_train))

    save_object(slice_df_and_sentences_list, "slice_df_and_sentences_list")
    return slice_df_and_sentences_list

def get_fold_by_index(n,k,i):
    slice_df_and_sentences_list = select_n_unique_items_k_times_knesset_sentences(n,k)
    return slice_df_and_sentences_list[i]





def select_and_prepare_emobank_sentences(vad_column="V"):

    df = pd.read_csv(emobank_csv_path, index_col=None)
    df = df[(df['text'].str.split().str.len() > 10) & (df['text'].str.split().str.len() < 30)]
    df = df[(df[f'norm-{str(vad_column)}'] < 0.3) | (df[f'norm-{str(vad_column)}'] > 0.7)]
    # df = df.sample(frac=0.25,random_state=42)

    print(f'emobank total sentences: {len(df)}')
    values = df[f'norm-{str(vad_column)}'].to_list()
    sentences = df['text'].to_list()

    sentences = [s.strip() for s in sentences]

    if LANGUAGE == Language.HEBREW:
        try:
            hebrew_sentences = load_object(f"{vad_column}-emobank_hebrew_sentences")
        except Exception as e:
            translator_model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt")
            translator_tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
            hebrew_sentences = []
            for sent in sentences:
                inputs = translator_tokenizer(f"<2he> {sent}", return_tensors="pt")
                outputs = translator_model.generate(**inputs, max_length=1024)
                # outputs = translator_model.generate(**inputs)
                translation = translator_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                hebrew_sentences.append(translation)
            save_object(hebrew_sentences,f"{vad_column}-emobank_hebrew_sentences")
        sentences = hebrew_sentences
    return sentences, values

def get_per_vad_dim_df(all_vad_df, hebrew=False):
    v_df = all_vad_df[['English Word', 'Valence']]
    a_df = all_vad_df[['English Word', 'Arousal']]
    d_df = all_vad_df[['English Word', 'Dominance']]
    return v_df,a_df,d_df
def remove_niqqud_from_string(my_string):
    vowel_pattern = re.compile(r"[\u0591-\u05C7]")
    try:
        new_string =  re.sub(vowel_pattern, "", my_string)
    except Exception as e:
        print(f'error:{e} in string: {my_string}')
        return my_string
    return new_string
def get_lexicon_words_from_df(all_lexicon_df, lang=Language.ENGLISH, with_nikud = True):
    if lang == Language.ENGLISH:
        words = list(all_lexicon_df['English Word'].values)
    elif lang == Language.HEBREW:
        words = list(all_lexicon_df['Hebrew Word'].values)
        if with_nikud:
            words = [remove_niqqud_from_string(word) for word in words]
    else:
        raise Exception("wrong language!")

    return words


def add_embeddings_to_vad_df(all_vad_df, embeddings):
    all_vad_df["english_word_embedding"] = list(embeddings)
    print(all_vad_df)

def create_and_train_regression_binom_model(x_train, y_train, model_name=None):
    binom_glm = sm.GLM(y_train, x_train, family=sm.families.Binomial())
    binom_model = binom_glm.fit()
    if model_name:
        save_object(binom_model, model_name)
    return  binom_model
def evaluate_model(model, x, y, data_name):
    y_preds = model.predict(x)
    # print(f'r^2 on {data_name} data:', metrics.r2_score(y, y_preds))
    res = pearsonr(y, y_preds)
    statistic = res.statistic
    pvalue = res.pvalue
    print(f"pearson's correlation  on {data_name} data: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def process_input_texts_for_multi_model(input_texts_list):
    new_inputs = [f'query: {text}' for text in input_texts_list]
    return new_inputs

def get_model_sentences_embeddings_from_last_hidden_state(sentences, model, tokenizer):
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_inputs)
    sentences_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    return sentences_embeddings


import torch


def get_gpt_style_sentence_embeddings_from_last_hidden_state(sentences, model, tokenizer):
    # Ensure the model is in evaluation mode
    model.eval()

    # Tokenize sentences and prepare input tensor
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move tensor to the same device as model
    inputs_on_device = {key: val.to(model.device) for key, val in encoded_inputs.items()}

    # Get model output with hidden states
    with torch.no_grad():
        model_output = model(**inputs_on_device, output_hidden_states=True)

    # Get the last hidden state
    last_hidden_state = model_output.hidden_states[-1]

    # Compute sentence embeddings by averaging all token embeddings in the last layer
    sentence_embeddings = last_hidden_state.mean(dim=1)

    # Move embeddings to CPU and convert to NumPy
    sentence_embeddings = sentence_embeddings.cpu().numpy()

    return sentence_embeddings


def get_alephbertgimmel_sentence_encoder_model():
    model_name = "imvladikon/alephbertgimmel-base-512"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def get_yam_peleg_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer
def get_auto_model_and_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
def get_dictabert_sentence_encoder_model():
    model_name = 'dicta-il/dictabert'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer



def create_full_hebrew_vad_lexicon_from_enriched_version(csv_enriched_lexicon_path, only_surface_forms=False):
    all_new_rows = []
    df = pd.read_csv(csv_enriched_lexicon_path)
    for index, row in df.iterrows():
        if str(row["NOT VALID"])!="nan":
            continue
        v = row["Valence"]
        a = row["Arousal"]
        d = row["Dominance"]
        hebrew_undotted = row["Hebrew Undotted"]
        if str(hebrew_undotted)=="nan":
            hebrew_undotted = row["Hebrew Word"]
        alternatives = []
        alternatives.append(hebrew_undotted)
        if not only_surface_forms:
            hebrew_lemma = row["Hebrew Lemma"]
            if str(hebrew_lemma)!="nan":
                alternatives.append(hebrew_lemma)
            for i in range(1, 8):
                word = row[f'Alternative {i}']
                if str(word)!= "nan":
                    alternatives.append(word)
        for word in alternatives:
            new_row = {}
            new_row["Valence"] = v
            new_row["Arousal"] = a
            new_row["Dominance"] = d
            new_row["Hebrew Word"] = word
            all_new_rows.append(new_row)
    new_vad_hebrew_df = pd.DataFrame.from_records(all_new_rows)
    if only_surface_forms:
        df_name = "vad_hebrew_only_surface_forms_lexicon.csv"
    else:
        df_name = "vad_full_hebrew_enriched_lexicon.csv"
    new_vad_hebrew_df.to_csv(df_name)
    return new_vad_hebrew_df

def check_lexicon_coverage_of_chosen_sentences(chosen_sentences_path, csv_vad_lexicon_path):
    vad_lex = pd.read_csv(csv_vad_lexicon_path)
    lex_words = list(vad_lex["Hebrew Word"])
    total_num_of_words_in_sentences = 0
    unique_words_in_sentences = set()
    unique_words_in_sentences_in_lexicon = set()
    num_of_sentences_words_in_lexicon = 0
    with open(chosen_sentences_path, encoding="utf-8") as file:
        for line in file.readlines():
            line_words = line.split()
            total_num_of_words_in_sentences +=len(line_words)
            for word in line_words:
                unique_words_in_sentences.add(word)
                if word in lex_words:
                    num_of_sentences_words_in_lexicon +=1
                    unique_words_in_sentences_in_lexicon.add(word)

    print(f'total number of words in sentences: {total_num_of_words_in_sentences}')
    print(f'number of sentences words in lexicon: {num_of_sentences_words_in_lexicon}')
    print(f'coverage percentage: {num_of_sentences_words_in_lexicon/total_num_of_words_in_sentences} ')
    print(f'number of unique words in sentences: {len(unique_words_in_sentences)} ')
    print(f'number of unique words in sentences which are in lexicon: {len(unique_words_in_sentences_in_lexicon)}')
    print(f'unique coverage: {len(unique_words_in_sentences_in_lexicon)/len(unique_words_in_sentences)}')


def load_our_fine_tuned_multi_model():
    global model_output_dir
    # Step 1: Load the original sentence transformer
    original_model_name = 'intfloat/multilingual-e5-large'
    original_model = SentenceTransformer(original_model_name)
    # Extract the original transformer backbone, pooling layer, and normalization layer
    transformer_backbone = original_model._modules['0']  # Transformer backbone
    pooling_layer = original_model._modules['1']  # Pooling layer
    normalize_layer = original_model._modules['2']  # Normalize layer
    # Step 2: Load the fine-tuned MLM model
    model_output_dir = "fine_tuned_on_dlc_server/fine-tuned-multi/fine_multi_checkpoint-134940"
    # fine_tuned_mlm_model = AutoModelForMaskedLM.from_pretrained(model_output_dir)
    # tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    fine_tuned_transformer_model = models.Transformer(
        model_name_or_path=model_output_dir,
        max_seq_length=512)
    # Step 3: Reassemble the SentenceTransformer model
    # Initialize a new SentenceTransformer model with the fine-tuned backbone
    modules = [fine_tuned_transformer_model, pooling_layer, normalize_layer]
    fine_tuned_model_reassembled = SentenceTransformer(modules=modules)
    return fine_tuned_model_reassembled


if __name__ == '__main__':
    our_multi_model = load_our_fine_tuned_multi_model()
    save_path = "fine_tuned_on_dlc_server/fine-tuned-multi/assembled_finetuned_multi_sentence_transformer"
    our_multi_model.save(save_path)
    csv_vad_lexicon_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\enriched_final_lexicon.csv"#TODO comment out this
    # create_full_hebrew_vad_lexicon_from_enriched_version(csv_vad_lexicon_path, only_surface_forms=True)
    if LANGUAGE == Language.HEBREW:
        # csv_vad_lexicon_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\vad_hebrew_only_surface_forms_lexicon.csv"
        csv_vad_lexicon_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\vad_full_hebrew_enriched_lexicon.csv"
    chosen_sentences_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism\\vad_manual_annotations_hebrew\\vad-sentence-annotations\\vad_poll_sentences.txt"
    check_lexicon_coverage_of_chosen_sentences(chosen_sentences_path, csv_vad_lexicon_path)
    if CONVERT_VAD_LEX_TO_CSV:
        convert_lexicon_file_to_csv(HEBREW_ENGLISH_VAD_LEXICON_PATH, output_path=csv_vad_lexicon_path)

    all_vad_df = pd.read_csv(csv_vad_lexicon_path)
    if LANGUAGE == Language.HEBREW:
        all_vad_df = all_vad_df.dropna(how='any')
    vad_words_list = get_lexicon_words_from_df(all_vad_df, lang=LANGUAGE, with_nikud=False)
    if TRAIN_WITH_EMO_BANK_DATA:
        v_emobank_sentences, v_emobank_values = select_and_prepare_emobank_sentences(vad_column="V")
        a_emobank_sentences, a_emobank_values = select_and_prepare_emobank_sentences(vad_column="A")
        d_emobank_sentences, d_emobank_values = select_and_prepare_emobank_sentences(vad_column="D")
    if TRAIN_WITH_KNESSET_DATA:
        fraction = 1 #todo
        knesset_sentences, knesset_vad_values_df = select_knesset_sentences(fraction=fraction)
        # knesset_sentences, knesset_vad_values_df = get_fold_by_index(96,5,4)#for folds only
        save_object(knesset_sentences, f'knesset_training_sentences')
        save_object(knesset_vad_values_df, f'knesset_training_sentences_vad_values_df')

    vad_embeddings_file_name = f"vad_word_sent_embeddings_{SENT_TRANSFORMER_TYPE}"
    v_emobank_embeddings_file_name = f'v_emobank_chosen_sentences_embeddings_{SENT_TRANSFORMER_TYPE}'
    a_emobank_embeddings_file_name = f'a_emobank_chosen_sentences_embeddings_{SENT_TRANSFORMER_TYPE}'
    d_emobank_embeddings_file_name = f'd_emobank_chosen_sentences_embeddings_{SENT_TRANSFORMER_TYPE}'
    knesset_sentences_embeddings_file_name = f'knesset_chosen_sentences_embeddings_{SENT_TRANSFORMER_TYPE}'
    if RECREATE_ENCODING_MODEL_AND_EMBEDDINGS:
        if ALEPHBERTGIMMEL:
            model_name = "imvladikon/alephbertgimmel-base-512"
            model, tokenizer = get_auto_model_and_tokenizer(model_name)
            vad_word_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(vad_words_list, model, tokenizer)
            save_object(vad_word_sent_embeddings, f"vad_word_sent_embeddings_alephbertgimmel")
        elif DICTABERT:
            model_name = 'dicta-il/dictabert'
            model, tokenizer = get_auto_model_and_tokenizer(model_name)
            vad_word_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(vad_words_list, model,
                                                                                             tokenizer)
            save_object(vad_word_sent_embeddings, f"vad_word_sent_embeddings_dictabert")
        elif YAM_PELEG:
            model_name = "yam-peleg/Hebrew-Mistral-7B"
            model, tokenizer = get_yam_peleg_model_and_tokenizer(model_name)
            # Process each sentence individually
            vad_word_sent_embeddings = []
            for word in vad_words_list:
                embedding = get_gpt_style_sentence_embeddings_from_last_hidden_state(word, model, tokenizer)
                vad_word_sent_embeddings.append(embedding)
            # vad_word_sent_embeddings = get_gpt_style_sentence_embeddings_from_last_hidden_state(vad_words_list, model,
            #                                                                                  tokenizer)
            save_object(vad_word_sent_embeddings, f"vad_word_sent_embeddings_yam-peleg")
        elif OUR_DICTA_PRETRAINED_MODEL:
            model_output_dir = "fine_tuned_on_dlc_server//model"
            # model_output_dir = "fine_tuned_on_dlc_server//checkpoint-11671"
            model, tokenizer = get_auto_model_and_tokenizer(model_output_dir)
            vad_word_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(vad_words_list, model,
                                                                                         tokenizer)
            save_object(vad_word_sent_embeddings, f"vad_word_sent_embeddings_fine_tuned_model")
        elif OUR_MULTI_PRETRAINED_MODEL:
            fine_tuned_model_reassembled = load_our_fine_tuned_multi_model()
            tokenizer = None

            vad_words_list = process_input_texts_for_multi_model(vad_words_list)
            if TRAIN_WITH_EMO_BANK_DATA:
                v_emobank_sentences = process_input_texts_for_multi_model(v_emobank_sentences)
                a_emobank_sentences = process_input_texts_for_multi_model(a_emobank_sentences)
                d_emobank_sentences = process_input_texts_for_multi_model(d_emobank_sentences)
            if TRAIN_WITH_KNESSET_DATA:
                knesset_sentences = process_input_texts_for_multi_model(knesset_sentences)
            model = fine_tuned_model_reassembled
            vad_word_sent_embeddings = model.encode(vad_words_list)
            save_object(vad_word_sent_embeddings, vad_embeddings_file_name)


        else:
            if MULTI:
                vad_words_list = process_input_texts_for_multi_model(vad_words_list)
                if TRAIN_WITH_EMO_BANK_DATA:
                    v_emobank_sentences = process_input_texts_for_multi_model(v_emobank_sentences)
                    a_emobank_sentences = process_input_texts_for_multi_model(a_emobank_sentences)
                    d_emobank_sentences = process_input_texts_for_multi_model(d_emobank_sentences)
                if TRAIN_WITH_KNESSET_DATA:
                    knesset_sentences = process_input_texts_for_multi_model(knesset_sentences)
            model = SentenceTransformer(SENT_TRANSFORMER_TYPE)
            vad_word_sent_embeddings = model.encode(vad_words_list)
            save_object(vad_word_sent_embeddings, vad_embeddings_file_name)

        if TRAIN_WITH_EMO_BANK_DATA:
            if DICTABERT or OUR_DICTA_PRETRAINED_MODEL:
                v_emo_bank_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(v_emobank_sentences, model,
                                                                                                 tokenizer)
                a_emo_bank_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(a_emobank_sentences,
                                                                                                   model,
                                                                                                   tokenizer)
                d_emo_bank_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(d_emobank_sentences,
                                                                                                   model,
                                                                                                   tokenizer)
            elif MULTI or OUR_MULTI_PRETRAINED_MODEL:
                v_emo_bank_sent_embeddings = model.encode(v_emobank_sentences)
                a_emo_bank_sent_embeddings = model.encode(a_emobank_sentences)
                d_emo_bank_sent_embeddings = model.encode(d_emobank_sentences)
            save_object(v_emo_bank_sent_embeddings, v_emobank_embeddings_file_name)
            save_object(a_emo_bank_sent_embeddings, a_emobank_embeddings_file_name)
            save_object(d_emo_bank_sent_embeddings, d_emobank_embeddings_file_name)
        if TRAIN_WITH_KNESSET_DATA:
            if DICTABERT or OUR_DICTA_PRETRAINED_MODEL:
                knesset_sentences_sent_embeddings = get_model_sentences_embeddings_from_last_hidden_state(knesset_sentences, model,
                                                                                                 tokenizer)
            elif MULTI or OUR_MULTI_PRETRAINED_MODEL:
                knesset_sentences_sent_embeddings = model.encode(knesset_sentences)
            save_object(knesset_sentences_sent_embeddings, knesset_sentences_embeddings_file_name)
    else:
        if ALEPHBERTGIMMEL:
            vad_word_sent_embeddings = load_object( f"vad_word_sent_embeddings_alephbertgimmel")
        elif DICTABERT:
            vad_word_sent_embeddings = load_object(f"vad_word_sent_embeddings_dictabert")
        elif YAM_PELEG:
            vad_word_sent_embeddings = load_object( f"vad_word_sent_embeddings_yam-peleg")
        elif OUR_DICTA_PRETRAINED_MODEL:
            vad_word_sent_embeddings = load_object(f"vad_word_sent_embeddings_fine_tuned_model")
        else:
            vad_word_sent_embeddings = load_object(vad_embeddings_file_name)
            if TRAIN_WITH_EMO_BANK_DATA:
                v_emo_bank_sent_embeddings = load_object(v_emobank_embeddings_file_name)
                a_emo_bank_sent_embeddings = load_object(a_emobank_embeddings_file_name)
                d_emo_bank_sent_embeddings = load_object(d_emobank_embeddings_file_name)
            if TRAIN_WITH_KNESSET_DATA:
                knesset_sentences_sent_embeddings = load_object(knesset_sentences_embeddings_file_name)

    X = vad_word_sent_embeddings

    if TRAIN_WITH_KNESSET_DATA:
        X_knesset = knesset_sentences_sent_embeddings
        X = np.concatenate((X, X_knesset), axis=0)

    if TRAIN_WITH_EMO_BANK_DATA:
        X_emo_v = v_emo_bank_sent_embeddings
        X_emo_a = a_emo_bank_sent_embeddings
        X_emo_d = d_emo_bank_sent_embeddings
        v_X = np.concatenate((X, X_emo_v), axis=0)
        a_X = np.concatenate((X, X_emo_a), axis=0)
        d_X = np.concatenate((X, X_emo_d), axis=0)
    else:
        v_X = X
        a_X = X
        d_X = X



    v_y = all_vad_df['Valence'].values
    a_y = all_vad_df['Arousal'].values
    d_y = all_vad_df['Dominance'].values

    if TRAIN_WITH_KNESSET_DATA:
        knesset_v_y = knesset_vad_values_df["v-normalized-score"]
        v_y = np.concatenate((v_y, knesset_v_y), axis=0)

        knesset_a_y = knesset_vad_values_df["a-normalized-score"]
        a_y = np.concatenate((a_y, knesset_a_y), axis=0)

        knesset_d_y = knesset_vad_values_df["d-normalized-score"]
        d_y = np.concatenate((d_y, knesset_d_y), axis=0)

    if TRAIN_WITH_EMO_BANK_DATA:
        emo_v_y = v_emobank_values
        v_y = np.concatenate((v_y, emo_v_y), axis=0)
        emo_a_y = a_emobank_values
        a_y = np.concatenate((a_y, emo_a_y), axis=0)
        emo_d_y = d_emobank_values
        d_y = np.concatenate((d_y, emo_d_y), axis=0)


    if USE_ALL_DATA_FOR_TRAINING:
        v_X_train = v_X
        a_X_train = a_X
        d_X_train = d_X
        v_y_train = v_y
        a_y_train = a_y
        d_y_train = d_y
    else:
        v_X_train, v_X_test, v_y_train, v_y_test = train_test_split(v_X, v_y, test_size = 1000, random_state = 42)
        a_X_train, a_X_test, a_y_train, a_y_test = train_test_split(a_X, a_y, test_size = 1000, random_state = 42)
        d_X_train, d_X_test, d_y_train, d_y_test = train_test_split(d_X, d_y, test_size = 1000, random_state = 42)
        assert(v_X_train.all() == a_X_train.all() == d_X_train.all())


    if RECREATE_AND_TRAIN_BINOM_REGRESSION_MODELS:
        create_and_train_regression_binom_model(v_X_train, v_y_train, model_name=Valence_regression_model_name)
        create_and_train_regression_binom_model(a_X_train, a_y_train, model_name=Arousal_regression_model_name)
        create_and_train_regression_binom_model(d_X_train, d_y_train, model_name=Dominance_regression_model_name)

    v_binom_model = load_object(Valence_regression_model_name)
    evaluate_model(v_binom_model, v_X_train, v_y_train, "Valence train")
    if not USE_ALL_DATA_FOR_TRAINING:
        evaluate_model(v_binom_model, v_X_test, v_y_test, "Valence test")

    a_binom_model = load_object(Arousal_regression_model_name)
    evaluate_model(a_binom_model, a_X_train, a_y_train, "Arousal train")
    if not USE_ALL_DATA_FOR_TRAINING:
        evaluate_model(a_binom_model, a_X_test, a_y_test, "Arousal test")

    d_binom_model = load_object(Dominance_regression_model_name)
    evaluate_model(d_binom_model, d_X_train, d_y_train, "Dominance train")
    if not USE_ALL_DATA_FOR_TRAINING:
        evaluate_model(d_binom_model, d_X_test, d_y_test, "Dominance test")




