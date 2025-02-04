import json
import os.path
import re
import statistics
import numpy as np
from wordfreq import word_frequency, tokenize, zipf_frequency
from lexicalrichness import LexicalRichness
import pandas as pd
from matplotlib import pyplot as plt
import pymannkendall as mk
from params_and_config import *
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

FIRST_YEAR_IN_CORPUS = 1992
LAST_YEAR_IN_CORPUS = 2022


def perform_t_test(series1, series2, series1_name= "series1", series2_name = "series2"):
    series1 = np.array(series1)
    series2 = np.array(series2)
    t_stat, p_value = stats.ttest_rel(series1, series2)

    mean_series1 = np.mean(series1)
    mean_series2 = np.mean(series2)

    print(f"t-statistic: {t_stat}, p-value: {p_value}")
    print(f"Mean of series1: {mean_series1}")
    print(f"Mean of series2: {mean_series2}")

    if p_value < 0.05:
        if mean_series1 > mean_series2:
            print(f"According to T-test the means are significantly different. {series1_name} has higher values.")
        else:
            print(f"According to T-test the means are significantly different. {series2_name}  has higher values.")
    else:
        print(f"The means of {series1_name} and {series2_name} are not significantly different.")


def calc_number_of_tokens_in_corpus(path_to_sentences_text_file):
    total_num_of_tokens = 0
    unique_tokens = set()
    with open(path_to_sentences_text_file, encoding="utf-8") as file:
        for line in file:
            sent_tokens = line.split()
            num_of_tokens_in_sent = len(sent_tokens)
            unique_tokens.update(sent_tokens)
            total_num_of_tokens += num_of_tokens_in_sent
    print(f"total number of tokens in corpus: {total_num_of_tokens}")
    print(f"number of unique tokens: {len(unique_tokens)}")

def calc_number_of_sentences_in_corpus(path_to_sentences_text_file):
    total_num_of_sentences = 0
    with open(path_to_sentences_text_file, encoding="utf-8") as file:
        for line in file:
            total_num_of_sentences +=1

    print(f"total number of sentences in corpus: {total_num_of_sentences}")
def calc_all_sentences_length(path):
    all_sents_length = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            sent_length = len(line.split())
            all_sents_length.append(sent_length)
    return all_sents_length

def calc_avg_length_of_knesset_sentences():
    all_sents_length = calc_all_sentences_length(os.path.join(knesset_txt_files_path, 'all_sentences_text.txt'))
    mean = statistics.mean(all_sents_length)
    std = statistics.stdev(all_sents_length)
    median = statistics.median(all_sents_length)
    print(f'The average sentence length is: {mean}')
    print(f'The standard deviation is {std} ')
    print(f'The median is {median} ')

    return all_sents_length

def calc_histogram(lengths_list):
    bins = {}
    for length in lengths_list:
        if length<=4:
            res = bins.get("01-04", 0)
            res += 1
            bins["01-04"] = res
        elif length<= 10:
            res = bins.get("05-10", 0)
            res +=1
            bins["05-10"] = res
        elif length<= 20:
            res = bins.get("11-20", 0)
            res +=1
            bins["11-20"] = res
        elif length <=30:
            res = bins.get("21-30", 0)
            res += 1
            bins["21-30"] = res
        elif length <=40:
            res = bins.get("31-40", 0)
            res += 1
            bins["31-40"] = res
        else:
            res = bins.get("41+", 0)
            res += 1
            bins["41+"] = res

    myKeys = list(bins.keys())
    myKeys.sort()
    sorted_bins = {i: bins[i] for i in myKeys}
    print(sorted_bins)
    return sorted_bins


def plot_histogram(bins_dict, title="histogram", rotation='horizontal', fig_size_0=None, fig_size_1 = None, ylim_min=-1, ylim_max=-1, y_ticks_min=-1, y_ticks_max=-1, y_ticks_step=-1):
    if fig_size_0 and fig_size_1:
        plt.figure(figsize=(fig_size_0, fig_size_1))
    plt.bar(bins_dict.keys(), bins_dict.values(), 1, color='deeppink', edgecolor=(0, 0, 0))
    if ylim_min>=0 and ylim_max>=0:
        plt.ylim(ylim_min, ylim_max)
    for i, v in enumerate(bins_dict.values()):
        plt.text(i - 0.45, v + 0.01, str(v), color='black', fontweight='bold', fontsize=12)
    if y_ticks_min>=0 and y_ticks_max>=0 and y_ticks_step>-0:
        plt.yticks(np.arange(y_ticks_min, y_ticks_max, y_ticks_step),fontsize=10)
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.show()

def get_only_first_half_year_sentences(csv_chunk, year, chair_seperation=False):
    if chair_seperation:
        start_year_df = csv_chunk[csv_chunk['protocol_date'].dt.strftime('%Y')==str(year)]
        end_year_df = csv_chunk[csv_chunk['protocol_date'].dt.strftime('%Y')==str(year+1)]
        first_half_of_start_year_df = start_year_df[start_year_df['protocol_date'].dt.strftime('%m').isin(WINTER_PERIOD_FIRST_YEAR_MONTHS)]
        first_half_of_end_year_df = end_year_df[end_year_df['protocol_date'].dt.strftime('%m').isin(WINTER_PERIOD_SECOND_YEAR_MONTHS)]
        first_half_of_year_df = pd.concat([first_half_of_start_year_df, first_half_of_end_year_df], ignore_index=True)
    else:
        year_df = csv_chunk[csv_chunk['protocol_date'].dt.strftime('%Y')==str(year)]
        first_half_of_year_df = year_df[year_df['protocol_date'].dt.strftime('%m').isin(FIRST_YEAR_MONTHS)]
    return first_half_of_year_df

def get_only_second_half_year_sentences(csv_chunk, year, chair_seperation=False):
    if chair_seperation:
        year_df = csv_chunk[csv_chunk['protocol_date'].dt.strftime('%Y')==str(year+1)]
        second_half_of_year_df = year_df[year_df['protocol_date'].dt.strftime('%m').isin(SUMMER_PERIOD_MONTHS)]
    else:
        year_df = csv_chunk[csv_chunk['protocol_date'].dt.strftime('%Y')==str(year)]
        second_half_of_year_df = year_df[year_df['protocol_date'].dt.strftime('%m').isin(SECOND_YEAR_MONTHS)]
    return second_half_of_year_df




def get_year_and_half_year_name(chair_seperation, half_year_num, year_num):
    if chair_seperation:
        if half_year_num == 0:
            half_year_name = "winter"
            year = year_num
        elif half_year_num == 1:
            half_year_name = "summer"
            year = year_num + 1
        else:
            print("what went wrong?")
    else:
        half_year_name = half_year_num
        year = year_num
    return half_year_name, year


def calc_type_to_token_ratio(txt_file):
    unique_tokens = set()
    total_num_of_tokens = 0
    with open(txt_file, encoding='utf-8') as file:
        for line in file:
            line_tokens = line.split()
            unique_tokens.update(line_tokens)
            total_num_of_tokens += len(line_tokens)
    print(f"ttr is {len(unique_tokens)/total_num_of_tokens}")
    print(f"total number of tokens: {total_num_of_tokens}")
    print(f"total number of unique tokens: {len(unique_tokens)}")



def plot_per_year_values(per_year_dict, title ="ttr per year",half_year=False, ylim_min=None, ylim_max=None,y_ticks_min=None,y_ticks_max=None,y_ticks_step=None, text_up=0.001,after_digit="%.3f"):
    years = list(per_year_dict.keys())
    values = list(per_year_dict.values())
    if half_year:
        years = [f'{years[i][0]}-{years[i][1]}' for i in range(len(years))]
    plt.figure(figsize=(100, 50))
    if ylim_min and ylim_max:
        plt.ylim(ylim_min, ylim_max)
    if y_ticks_min and y_ticks_max and y_ticks_step:
        plt.yticks(np.arange(y_ticks_min, y_ticks_max, y_ticks_step),fontsize=60)
    else:
        plt.yticks(fontsize=60)
    plt.xticks(fontsize=50, rotation='vertical')


    plt.bar(range(len(per_year_dict)), values, width=0.8, tick_label=years, color='deeppink', edgecolor=(0, 0, 0))
    plt.title(title, fontsize = 80)
    for i, v in enumerate(values):
        plt.text(i - 0.45, v + text_up, str(after_digit % v),color = 'black', fontweight = 'bold',fontsize=45)

    plt.show()
    print("done!")


def create_year_score_dict_from_file(path):
    per_year_ttr = {}
    with open(path) as f:
        for line in f:
           year = line.split("in year ")[1].split()[0]
           score = float(line.split(" is ")[1].split()[0][:-1])
           per_year_ttr[year] = score
    return per_year_ttr

def create_half_year_score_dict_from_file(path):
    per_year_ttr = {}
    with open(path) as f:
        for line in f:
           year = line.split("in year ")[1].split()[0]
           part = line.split("part ")[1].split()[0]
           score = float(line.split(" is ")[1].split()[0][:-1])
           per_year_ttr[(year,part)] = score
    return per_year_ttr



def how_many_protocols_per_year(path_to_protocols, protocol_type="committee"):
    knesset_nums = os.listdir(path_to_protocols)
    for knesset_num in knesset_nums:
        word_year_dirs_path = os.path.join(path_to_protocols,knesset_num,"word_files")
        years_dirs = os.listdir(word_year_dirs_path)
        for year in years_dirs:
            year_protocols_path = os.path.join(word_year_dirs_path, year)
            num_of_protocols_in_year = len(os.listdir(year_protocols_path))
            print(f'The number of {protocol_type} protocols in knesset {knesset_num} in year {year} is: {num_of_protocols_in_year}')


def apply_Mann_Kendall_Test(data, significance_level=0.05, data_name=""):
    res = mk.original_test(data)
    print(f'p_value is: {res.p}')
    if res.p < significance_level:
        print(
            f'the p-value of the test is lower than significance level = {significance_level} so there is statistically significant evidence that a trend is present in the time series data: {data_name}')
        print(f'the trend is: {res.trend}\n')
    else:
        print(f'the p-value of the test is higher than significance level in the time series data: {data_name}\n')

def calculate_number_of_sentences_in_sentences_jsonl_file(jsonl_file):
    num_of_sentences = 0
    with open(jsonl_file,encoding="utf-8") as file:
        for line in file:
            num_of_sentences +=1
    print(f' num of sentences in {os.path.basename(jsonl_file)} is : {num_of_sentences}')
    return num_of_sentences



