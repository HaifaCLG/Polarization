from glob import glob

import transformers
import inspect

import os.path
import random
import re

import pandas as pd
import scipy
import torch

from datasets import Dataset, DatasetDict
from scipy.stats import pearsonr
from enum import Enum
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import cos_sim

import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput

from aux_functions import *
from aux_functions import *
random.seed(42)
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
from safetensors.torch import load_file as load_safetensors
import sys
sys.stdout.reconfigure(line_buffering=True)

print("🤖 transformers version:", transformers.__version__, flush=True)
print("🧩 TrainingArguments lives in:", TrainingArguments.__module__, flush=True)
VAD_TRAINING_DATA = "/data"
HEBREW_ENGLISH_VAD_LEXICON_PATH = os.path.join(data_path,"vad","NRC-VAD-Lexicon","NRC-VAD-Lexicon","OneFilePerLanguage","Hebrew-NRC-VAD-Lexicon.txt")

class Language(Enum):
    ENGLISH = "english"
    HEBREW = "hebrew"


FREEZE = True
LANGUAGE = Language.HEBREW
if LANGUAGE== Language.HEBREW:
    SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large'
    # SENT_TRANSFORMER_TYPE = 'imvladikon/sentence-transformers-alephbert'
    # SENT_TRANSFORMER_TYPE = "imvladikon/alephbertgimmel-base-512"
    # SENT_TRANSFORMER_TYPE = 'dicta-il/dictabert'
    # SENT_TRANSFORMER_TYPE = "yam-peleg/Hebrew-Mistral-7B"
else:
    # SENT_TRANSFORMER_TYPE = 'bert-base-nli-mean-tokens'
    SENT_TRANSFORMER_TYPE = 'bert-large-nli-mean-tokens'
    # SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large'

if SENT_TRANSFORMER_TYPE == 'intfloat/multilingual-e5-large':
    MULTI = False# restore this for original multi
else:
    MULTI = False

ALEPHBERTGIMMEL = False
DICTABERT = False
YAM_PELEG = False
OUR_DICTA_PRETRAINED_MODEL = False
OUR_MULTI_PRETRAINED_MODEL = True# change to false for original multi

if OUR_DICTA_PRETRAINED_MODEL:
    SENT_TRANSFORMER_TYPE = 'dicta-il/dictabert-ourfinetuned'


if OUR_MULTI_PRETRAINED_MODEL:
    SENT_TRANSFORMER_TYPE = 'intfloat/multilingual-e5-large-ourfinetuned'

CONVERT_VAD_LEX_TO_CSV = False#should happen only once
RECREATE_ENCODING_MODEL_AND_EMBEDDINGS = True
RECREATE_AND_TRAIN_BINOM_REGRESSION_MODELS = True
USE_ALL_DATA_FOR_TRAINING = False# change back to true for original multi
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
    knesset_poll_sentences_path = f"{VAD_TRAINING_DATA}/knesset_poll_sentences_avg_scores_all_vad.csv"
    df = pd.read_csv(knesset_poll_sentences_path, index_col=None)
    df = df.sample(frac=fraction, random_state=42)
    print(f'knesset total sentences: {len(df)}', flush=True)
    sentences = df["sent_text"].to_list()

    return sentences, df

def load_best_checkpoint(output_dir):
    # List all checkpoint sub‐dirs
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {output_dir}")
    # Sort them (they’re named checkpoint-1, checkpoint-2, …)
    latest = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]
    return latest
def select_n_unique_items_k_times_knesset_sentences(n_train=96, k=5):
    output_path = os.path.join('pickle_objects', f'{"slice_df_and_sentences_list"}.pkl')
    if os.path.exists(output_path):
        slice_df_and_sentences_list = load_object("slice_df_and_sentences_list")
        return slice_df_and_sentences_list
    knesset_poll_sentences_path = f"{VAD_TRAINING_DATA}/knesset_poll_sentences_avg_scores_all_vad.csv"
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


def find_model_file(root):
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    # fallback: pick the last checkpoint directory
    ckpts = sorted(glob(os.path.join(root, "checkpoint-*")))
    for ck in reversed(ckpts):
        for name in ("model.safetensors", "pytorch_model.bin"):
            path = os.path.join(ck, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(f"No model file in {root} or its checkpoints")


def select_and_prepare_emobank_sentences(vad_column="V"):
    emobank_csv_path = f"{VAD_TRAINING_DATA}/emobank.csv"
    df = pd.read_csv(emobank_csv_path, index_col=None)
    df = df[(df['text'].str.split().str.len() > 10) & (df['text'].str.split().str.len() < 30)]
    df = df[(df[f'norm-{str(vad_column)}'] < 0.3) | (df[f'norm-{str(vad_column)}'] > 0.7)]
    # df = df.sample(frac=0.25,random_state=42)

    print(f'emobank total sentences: {len(df)}', flush=True)
    values = df[f'norm-{str(vad_column)}'].to_list()
    sentences = df['text'].to_list()

    sentences = [s.strip() for s in sentences]

    if LANGUAGE == Language.HEBREW:
        try:
            hebrew_sentences = load_object(f"{vad_column}-emobank_hebrew_sentences")
        except Exception as e:
            print(f"Couldnt find pickle object to load {vad_column}-emobank_hebrew_sentences", flush=True)
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
        print(f'error:{e} in string: {my_string}', flush=True)
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
    print(all_vad_df, flush=True)

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
    print(f"pearson's correlation  on {data_name} data: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}", flush=True)

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

    print(f'total number of words in sentences: {total_num_of_words_in_sentences}', flush=True)
    print(f'number of sentences words in lexicon: {num_of_sentences_words_in_lexicon}', flush=True)
    print(f'coverage percentage: {num_of_sentences_words_in_lexicon/total_num_of_words_in_sentences} ', flush=True)
    print(f'number of unique words in sentences: {len(unique_words_in_sentences)} ', flush=True)
    print(f'number of unique words in sentences which are in lexicon: {len(unique_words_in_sentences_in_lexicon)}', flush=True)
    print(f'unique coverage: {len(unique_words_in_sentences_in_lexicon)/len(unique_words_in_sentences)}', flush=True)


def load_our_fine_tuned_multi_model():
    global model_output_dir
    # Load the original sentence transformer
    original_model_name = 'intfloat/multilingual-e5-large'
    original_model = SentenceTransformer(original_model_name)
    # Extract the original transformer backbone, pooling layer, and normalization layer
    transformer_backbone = original_model._modules['0']  # Transformer backbone
    pooling_layer = original_model._modules['1']  # Pooling layer
    normalize_layer = original_model._modules['2']  # Normalize layer
    #  Load the fine-tuned MLM model
    model_output_dir = "/app/fine_tuned_on_dlc_server/fine-tuned-multi/fine_multi_checkpoint-134940"
    model_output_dir = os.path.abspath(model_output_dir)
    fine_tuned_transformer_model = models.Transformer(
        model_name_or_path=model_output_dir,
        max_seq_length=512)
    # Reassemble the SentenceTransformer model
    # Initialize a new SentenceTransformer model with the fine-tuned backbone
    modules = [fine_tuned_transformer_model, pooling_layer, normalize_layer]
    fine_tuned_model_reassembled = SentenceTransformer(modules=modules)
    return fine_tuned_model_reassembled

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    K_FOLD = True
    fold_i = 3
    if K_FOLD:
        print(f'working on fold {fold_i}', flush=True)
    our_multi_model = load_our_fine_tuned_multi_model()
    save_path = "/app/fine_tuned_on_dlc_server/fine-tuned-multi/assembled_finetuned_multi_sentence_transformer"
    if LANGUAGE == Language.HEBREW:
        csv_vad_lexicon_path = f"{VAD_TRAINING_DATA}/vad_full_hebrew_enriched_lexicon.csv"
    chosen_sentences_path = f"{VAD_TRAINING_DATA}/vad-sentence-annotations/vad_poll_sentences.txt"
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
        fraction = 0.7
        knesset_sentences, knesset_vad_values_df = select_knesset_sentences(fraction=fraction)
        if K_FOLD:
            knesset_sentences, knesset_vad_values_df = get_fold_by_index(96,5,fold_i)#for folds only
            save_object(knesset_sentences, f'knesset_training_sentences_fold_{fold_i}')
            save_object(knesset_vad_values_df, f'knesset_training_sentences_vad_values_df_fold_{fold_i}')
        save_object(knesset_sentences, f'knesset_training_sentences')
        save_object(knesset_vad_values_df, f'knesset_training_sentences_vad_values_df')


    # if RECREATE_ENCODING_MODEL_AND_EMBEDDINGS:
    if ALEPHBERTGIMMEL:
        model_name = "imvladikon/alephbertgimmel-base-512"
        model, tokenizer = get_auto_model_and_tokenizer(model_name)
    elif DICTABERT:
        model_name = 'dicta-il/dictabert'
        model, tokenizer = get_auto_model_and_tokenizer(model_name)
    elif OUR_DICTA_PRETRAINED_MODEL:
        model_output_dir = "/app/fine_tuned_on_dlc_server/model"
        model_name = "/app/fine_tuned_on_dlc_server/model"
        # model_output_dir = "fine_tuned_on_dlc_server//checkpoint-11671"
        model, tokenizer = get_auto_model_and_tokenizer(model_output_dir)

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
        our_multi_encoder_model_dir = "/app/fine_tuned_on_dlc_server/fine-tuned-multi/fine_multi_checkpoint-134940"
        model_name = our_multi_encoder_model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    else:
        if MULTI:
            vad_words_list = process_input_texts_for_multi_model(vad_words_list)
            if TRAIN_WITH_EMO_BANK_DATA:
                v_emobank_sentences = process_input_texts_for_multi_model(v_emobank_sentences)
                a_emobank_sentences = process_input_texts_for_multi_model(a_emobank_sentences)
                d_emobank_sentences = process_input_texts_for_multi_model(d_emobank_sentences)
            if TRAIN_WITH_KNESSET_DATA:
                knesset_sentences = process_input_texts_for_multi_model(knesset_sentences)
            model_name = 'intfloat/multilingual-e5-large'
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = SentenceTransformer(SENT_TRANSFORMER_TYPE)

        # vad_word_sent_embeddings = model.encode(vad_words_list)
        # save_object(vad_word_sent_embeddings, vad_embeddings_file_name)



    X = vad_words_list

    if TRAIN_WITH_KNESSET_DATA:
        X += knesset_sentences

    if TRAIN_WITH_EMO_BANK_DATA:
        X_emo_v = v_emobank_sentences
        X_emo_a = a_emobank_sentences
        X_emo_d = d_emobank_sentences
        v_X = X.copy()
        v_X += X_emo_v
        a_X = X.copy()
        a_X +=X_emo_a
        d_X = X.copy()
        d_X += X_emo_d
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


    def prepare_dataset(texts, scores, test_size=0.2, seed=42):
        ds = Dataset.from_dict({"text": texts, "labels": scores})
        if USE_ALL_DATA_FOR_TRAINING:
            return DatasetDict({"train": ds})
        else:
            split = ds.train_test_split(test_size=test_size, seed=seed)
            return DatasetDict({
                "train": split["train"],
                "validation": split["test"]
            })


    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )


    def make_model():
        print(f'model name for training: {model_name}', flush=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        )
        # Freeze all backbone layers
        if FREEZE:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        return model


    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.squeeze()
        pearson_r, pval = pearsonr(labels, preds)
        return {"pearson": pearson_r, "pearson_p": pval}


    def train_and_eval(sentences, scores, output_dir):
        ds_dict = prepare_dataset(sentences, scores)
        # tokenize
        tokenized = ds_dict.map(tokenize_batch, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # args
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="pearson",
            greater_is_better=True,
            logging_steps=10,
            report_to=["none"],
        )
        trainer = Trainer(
            model=make_model(),
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized.get("validation", None),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        print(f'Started training to {output_dir}', flush=True)
        trainer.train()
        print(f'finished training. Now saving model to {output_dir}', flush=True)
        trainer.save_model(output_dir)
        print(f'model saved to {output_dir}', flush=True)

        print(trainer.evaluate(), flush=True)
        return tokenized, trainer


    def evaluate_on_sentences(sentences, labels, model_dir, tokenizer):
        if OUR_MULTI_PRETRAINED_MODEL:
            sentences = process_input_texts_for_multi_model(sentences)
        ds = Dataset.from_dict({"text": sentences, "labels": labels})
        ds = ds.map(tokenize_batch, batched=True)
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            print(f'loading from checkpoint')
            ckpt_dir = find_model_file(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir,num_labels=1,problem_type="regression")

        # create a Trainer *without* train_dataset
        args = TrainingArguments(
            output_dir="eval_temp",
            do_train=False,
            per_device_eval_batch_size=32,
            report_to=["none"],
        )
        trainer = Trainer(
            model=model,
            args=args,
            tokenizer=tokenizer,
            eval_dataset=ds
        )

        preds = trainer.predict(ds).predictions.squeeze()
        true_labels  = trainer.predict(ds).label_ids

        r, p = pearsonr(preds, true_labels )
        print(f"Pearson-r = {r:.3f}, p = {p:.3f}", flush=True)



    RETRAIN_VAD_MODELS = True
    model_print_name = model_name

    if OUR_DICTA_PRETRAINED_MODEL:
        model_print_name = "knesset-dicta"
    elif OUR_MULTI_PRETRAINED_MODEL:
        model_print_name = "Knesset-multi-assembled"

    if FREEZE:
        model_print_name = f'frozen_backbone_{model_print_name}'
    else:
        model_print_name = f'unfrozen_{model_print_name}'
    if K_FOLD:
        fold_name = f'_fold_{fold_i}'
    else:
        fold_name = ""
    valence_model_dir = f"/app/valence-regression_head_{model_print_name.replace('/', '')}{fold_name}"
    arousal_model_dir = f"/app/arousal-regression_head_{model_print_name.replace('/', '')}{fold_name}"
    dominance_model_dir = f"/app/dominance-regression_head_{model_print_name.replace('/', '')}{fold_name}"
    if RETRAIN_VAD_MODELS:
        print(f'training V:', flush=True)
        _, v_trainer = train_and_eval(v_X, v_y, valence_model_dir)
        print(f'Trainer info:', flush=True)
        print(v_trainer, flush=True)
        print(f'training A:', flush=True)
        _, a_trainer = train_and_eval(a_X, a_y, arousal_model_dir)
        print(f'Trainer info:', flush=True)
        print(a_trainer, flush=True)
        print(f'training D:', flush=True)
        _, d_trainer = train_and_eval(d_X, d_y, dominance_model_dir)
        print(d_trainer, flush=True)
    else:
        all_sents, all_df = select_knesset_sentences(1)
        if K_FOLD:
            trained_sents = load_object(f'knesset_training_sentences_vad_values_df_fold_{fold_i}')
        else:
            trained_sents = load_object("knesset_training_sentences")

        test_sents = [s for s in all_sents if s not in trained_sents]

        # align DataFrame to get true values in order
        test_df = (
            all_df[all_df.sent_text.isin(test_sents)]
            .set_index("sent_text")
            .loc[test_sents]
            .reset_index()
        )
        v_labels = test_df["v-normalized-score"].tolist()
        a_labels = test_df["a-normalized-score"].tolist()
        d_labels = test_df["d-normalized-score"].tolist()

        # Call the evaluator for each dimension:
        print("Valence test results:", flush=True)
        evaluate_on_sentences(test_sents, v_labels, valence_model_dir, tokenizer)
        print(f'A model dir: {arousal_model_dir}', flush=True)
        print("Arousal test results:", flush=True)
        evaluate_on_sentences(test_sents, a_labels, arousal_model_dir, tokenizer)
        print(f'D model dir: {dominance_model_dir}', flush=True)
        print("Dominance test results:", flush=True)
        evaluate_on_sentences(test_sents, d_labels, dominance_model_dir, tokenizer)









