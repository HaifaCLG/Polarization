import json
import os.path

import pandas as pd
from params_and_config import *
from aux_functions import *
from statistic_functions import FIRST_YEAR_IN_CORPUS, LAST_YEAR_IN_CORPUS, get_only_first_half_year_sentences, \
    get_only_second_half_year_sentences, get_year_and_half_year_name, apply_Mann_Kendall_Test

from datetime import datetime
from contextlib import redirect_stdout



def calc_protocol_sum_of_attribute_grades_and_num_of_lexicon_words(protocol_path, lexicon_grade_dict):
    sum_of_grades = 0
    num_of_graded_words = 0
    total_num_of_tokens_in_protocol = 0
    try:
        with open(protocol_path, encoding="utf-8") as file:
            protocol_json = json.load(file)
    except Exception as e:
        raise Exception(f'couldnt load protocol json: {protocol_path}. error was: {e}')

    for sentence in protocol_json["protocol_sentences"]:
        features = sentence["morphological_fields"]
        if features:
            for token_item in features:
                if isinstance(token_item["id"], list) and "-" in token_item["id"]:
                    continue
                total_num_of_tokens_in_protocol += 1
                word = token_item["form"]
                lemma = token_item["lemma"]
                lemma_grade = lexicon_grade_dict.get(lemma, -1)
                if lemma_grade != -1:
                    sum_of_grades += lemma_grade
                    num_of_graded_words += 1
                else:
                    word_grade = lexicon_grade_dict.get(word, -1)
                    if word_grade != -1:
                        sum_of_grades += word_grade
                        num_of_graded_words += 1
        else:
            sent_tokens = sentence["sentence_text"].split()
            for token in sent_tokens:
                total_num_of_tokens_in_protocol += 1
                word = token
                word_grade = lexicon_grade_dict.get(word, -1)
                if word_grade != -1:
                    sum_of_grades += word_grade
                    num_of_graded_words += 1


    return sum_of_grades, num_of_graded_words, total_num_of_tokens_in_protocol





def get_only_non_verb_parsings_from_lexicon_lemma_file(lemma_conllu_file, only_non_verb_lemmas_output):
    text_to_write = ""
    with open(lemma_conllu_file, encoding="utf-8") as file:
        data = file.read()
    conllu_sents = parse_each_conllu_sentence_separatly(data)
    non_verb_conllu_sents = []
    for conllu_sent in conllu_sents:
        if len(conllu_sent)==3:
            if conllu_sent[2]['xpos'] != "VERB":
                non_verb_conllu_sents.append(conllu_sent)
        elif len(conllu_sent) == 4:
            if conllu_sent[3]['xpos'] != "VERB":
                non_verb_conllu_sents.append(conllu_sent)
        else:
            non_verb_conllu_sents.append(conllu_sent)
    print(len(non_verb_conllu_sents))
    words = set()
    for sent in non_verb_conllu_sents:
        sent_text = sent.metadata["text"]
        word = sent_text.split()[2]
        words.add(word)
        serialized = sent.serialize()
        text_to_write += serialized
        text_to_write+="\n"
    with open(only_non_verb_lemmas_output, "w", encoding="utf-8" ) as file:
        file.write(text_to_write)
    for word in words:
        print(word)


def calc_lexicon_word_attribute_grade_dict(attribute, lexicon_path, word_grade_dict_name):
    word_grade_dict = {}
    duplicate_words_grades_dict = {}
    lexicon_df = pd.read_csv(lexicon_path, keep_default_na=False)
    lexicon_df = lexicon_df.dropna(how='any')
    alternative_hebrew_columns = ["Hebrew Lemma", "Alternative 1", "Alternative 2", "Alternative 3", "Alternative 4", "Alternative 5", "Alternative 6", "Alternative 7"]
    for index, row in lexicon_df.iterrows():
        if row["NOT VALID"].strip() != "":
            continue
        if row["Hebrew Undotted"].strip():
            word = row["Hebrew Undotted"].strip()
        else:
            word = row["Hebrew Word"].strip()
        grade = word_grade_dict.get(word, -1)
        if grade == -1:
            word_grade_dict[word] = row[attribute]
        else:
            update_duplcate_words_grades_dict(attribute, duplicate_words_grades_dict, grade, row, word)

        for column in alternative_hebrew_columns:
            if row[column].strip():
                word = row[column].strip()
                grade = word_grade_dict.get(word, -1)
                if grade == -1:
                    word_grade_dict[word] = row[attribute]
                else:
                    update_duplcate_words_grades_dict(attribute, duplicate_words_grades_dict, grade, row, word)
    for word, grade_list in duplicate_words_grades_dict.items():
        grade_sum = sum(grade_list)
        num = len(grade_list)
        avg_grade = grade_sum/num
        word_grade_dict[word] = avg_grade
    save_object(word_grade_dict, word_grade_dict_name)
    return word_grade_dict




def update_dict_with_fixed_lemma(attribute):
    word_grade_dict = load_object(f"{attribute}_word_grade_dict")
    for word, grade in word_grade_dict.items():
        pass

def update_duplcate_words_grades_dict(attribute, duplicate_words_grades_dict, grade, row, word):
    grade_list = duplicate_words_grades_dict.get(word, [])
    if grade_list:
        grade_list.append(row[attribute])
    else:
        grade_list.append(grade)
        grade_list.append(row[attribute])
    duplicate_words_grades_dict[word] = grade_list


def calc_attribute_sum_of_grades_and_num_of_lexicon_words_per_protocol(protocols_path, lexicon_grade_dict,
                                                                       output_dict_name, per_protocol_graded_words_out_of_total_words_dict_name):
    protocol_attribute_dict = {}
    protocol_graded_words_out_of_total_words_dict = {}
    protocol_names = os.listdir(protocols_path)
    for protocol_name in protocol_names:
        protocol_path = os.path.join(protocols_path, protocol_name)
        try:
            sum_of_grades, num_of_lexicon_words, total_num_of_tokens_in_protocol = calc_protocol_sum_of_attribute_grades_and_num_of_lexicon_words(protocol_path, lexicon_grade_dict)
        except Exception as e:
            print(e)
            continue
        protocol_attribute_dict[protocol_name.split(".jsonl")[0]] = (sum_of_grades, num_of_lexicon_words)
        protocol_graded_words_out_of_total_words_dict[protocol_name.split(".jsonl")[0]] = (num_of_lexicon_words, total_num_of_tokens_in_protocol)

    save_object(protocol_attribute_dict, output_dict_name)
    save_object(protocol_graded_words_out_of_total_words_dict, per_protocol_graded_words_out_of_total_words_dict_name)

def create_a_protocol_name_and_date_csv_file_from_protocols_jsonl_file(protocols_jsonl_file, protocol_name_and_date_csv_file_path):
    data = []
    with open(protocols_jsonl_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                json_line = json.loads(line)
                filtered_data = {k: json_line[k] for k in
                                 ('protocol_name', 'protocol_date')}
                data.append(filtered_data)
            except Exception as e:
                print(f'couldnt load json. error was:{e}')
                continue

    df = pd.DataFrame(data)
    df.to_csv(protocol_name_and_date_csv_file_path, index=False)
def calc_attribute_frequency_per_chair(protocol_name_and_date_csv_file_path, attribute, dict_name, per_protocol_graded_words_out_of_total_words_dict_name, per_chair_dict_name):
    per_chair_attribute_frequency_dict = {}
    per_chair_lexicon_words_frequency = {}
    attribute_dict = load_object(dict_name)
    chunk_container = pd.read_csv(protocol_name_and_date_csv_file_path, chunksize=1000)
    per_chair_attribute_sum_of_grades_dict = {}
    per_chair_attribute_total_number_of_lexicon_words_dict = {}
    per_chair_total_num_of_tokens_dict = {}
    protocol_graded_words_out_of_total_words_dict = load_object(per_protocol_graded_words_out_of_total_words_dict_name)
    per_chair_num_lexicon_words_dict = {}
    for protocols_df in chunk_container:
        protocols_df["protocol_date"] = pd.to_datetime(protocols_df['protocol_date'])
        for year_num in range(FIRST_YEAR_IN_CORPUS - 1,
                              LAST_YEAR_IN_CORPUS):  # This algorithm adds 1 to year for summer months
            first_half_of_year_df = get_only_first_half_year_sentences(protocols_df, year_num,
                                                                       chair_seperation=True)
            second_half_of_year_df = get_only_second_half_year_sentences(protocols_df, year_num,
                                                                         chair_seperation=True)
            two_halfs_of_year = [first_half_of_year_df, second_half_of_year_df]
            for half_year_df, half_year_num in zip(two_halfs_of_year, range(len(two_halfs_of_year))):
                half_year_name, year = get_year_and_half_year_name(True, half_year_num, year_num)
                for index, row in half_year_df.iterrows():
                    chair_sum_of_grades = per_chair_attribute_sum_of_grades_dict.get((year, half_year_name), 0)
                    chair_total_number_of_lexicon_words = per_chair_attribute_total_number_of_lexicon_words_dict.get((year, half_year_name), 0)
                    chair_num_of_lexicon_words = per_chair_num_lexicon_words_dict.get((year, half_year_name), 0)
                    chair_total_num_of_tokens = per_chair_total_num_of_tokens_dict.get((year, half_year_name), 0)
                    protocol_name = row["protocol_name"]
                    protocol_sum_of_grades, protocol_num_of_graded_tokens = attribute_dict.get(protocol_name, (0,0))
                    protocol_num_of_lexicon_words, total_num_of_tokens_in_protocol = protocol_graded_words_out_of_total_words_dict.get(protocol_name, (0,0))
                    if protocol_num_of_lexicon_words != protocol_num_of_graded_tokens:
                        print(f'not same number of graded words in dictionaries: first:{protocol_num_of_graded_tokens}, second: {protocol_num_of_lexicon_words}')
                    chair_num_of_lexicon_words += protocol_num_of_lexicon_words
                    chair_total_num_of_tokens += total_num_of_tokens_in_protocol
                    per_chair_num_lexicon_words_dict[(year, half_year_name)] = chair_num_of_lexicon_words
                    per_chair_total_num_of_tokens_dict[(year, half_year_name)] = chair_total_num_of_tokens
                    chair_sum_of_grades += protocol_sum_of_grades
                    chair_total_number_of_lexicon_words += protocol_num_of_graded_tokens
                    per_chair_attribute_sum_of_grades_dict[(year, half_year_name)] = chair_sum_of_grades
                    per_chair_attribute_total_number_of_lexicon_words_dict[(year, half_year_name)] = chair_total_number_of_lexicon_words


    for year_num in range(FIRST_YEAR_IN_CORPUS - 1, LAST_YEAR_IN_CORPUS):
        for half_year_num in range(len(two_halfs_of_year)):
            half_year_name, year = get_year_and_half_year_name(True, half_year_num, year_num)
            chair_sum_of_grades = per_chair_attribute_sum_of_grades_dict.get((year, half_year_name), 0)
            chair_total_number_of_lexicon_words = per_chair_attribute_total_number_of_lexicon_words_dict.get(
                (year, half_year_name), 0)
            if chair_total_number_of_lexicon_words != 0:
                per_chair_attribute_frequency_dict[(year, half_year_name)] = chair_sum_of_grades/ chair_total_number_of_lexicon_words
            else:
                print(f'no lexicon words in chair {year}-{half_year_name}')
                continue
            chair_num_of_lexicon_words = per_chair_num_lexicon_words_dict.get(
                (year, half_year_name), 0)
            chair_total_num_of_tokens = per_chair_total_num_of_tokens_dict.get(
                (year, half_year_name), 0)
            if chair_total_num_of_tokens != 0:
                per_chair_lexicon_words_frequency[(year, half_year_name)] = chair_num_of_lexicon_words/chair_total_num_of_tokens
            else:
                print(f'no words in chair {year}-{half_year_name}')
                continue
    save_object(per_chair_attribute_frequency_dict, per_chair_dict_name)
    save_object(per_chair_lexicon_words_frequency, "per_chair_lexicon_words_frequency_dict")

    return per_chair_attribute_frequency_dict

def enrich_lexicon_grade_dict_with_lemmas_from_trankit_output(original_dict_name, conllu_file, new_lexicon_grade_dict_name):
    grade_dict = load_object(original_dict_name)
    grade_dict.pop("")
    with open(conllu_file, encoding="utf-8") as file:
        data = file.read()
    conllu_words = parse_each_conllu_sentence_separatly(data)
    for conllu_word in conllu_words:
        if len(conllu_word)> 1:
            if isinstance(conllu_word[0]["id"], tuple) and "-" in conllu_word[0]["id"] and conllu_word[1]["form"] == "ל" and conllu_word[2]["xpos"]== "VERB":
                lemma = conllu_word[2]["lemma"]
                original_word = conllu_word[0]["form"]
                print(f"original word: {original_word} lemma: {lemma}")
            else:
                continue
        else:
            if conllu_word[0]["form"][0] == "ל" and conllu_word[0]["xpos"] == "VERB":
                original_word = conllu_word[0]["form"]
                lemma = conllu_word[0]["lemma"]
                print(f"original word: {original_word} lemma: {lemma}")
            else:
                continue


        original_word_grade = grade_dict.get(original_word, -1)
        if original_word_grade == -1:
            print(f'error: original word not in dict')
        lemma_grade = grade_dict.get(lemma, -1)
        if lemma_grade == -1:
            grade_dict[lemma] = original_word_grade
        else:
            continue
    save_object(grade_dict,new_lexicon_grade_dict_name)

def add_vad_hebrew_annotations_to_emo_lexicon(vad_lexicon_path, csv_emotion_lexicon_path, new_lexicon_output_path):
    merged_rows = []
    emotions_attributes = ["anger","anticipation","disgust","fear","joy","negative","positive","sadness","surprise","trust"]
    vad_df = pd.read_csv(vad_lexicon_path, keep_default_na=False)
    vad_hebrew_columns = ["Hebrew Word","Hebrew Undotted","Hebrew Lemma","Alternative 1","Alternative 2","Alternative 3","Alternative 4","Alternative 5","Alternative 6","Alternative 7","NOT VALID"]
    emo_df = pd.read_csv(csv_emotion_lexicon_path, keep_default_na=False)
    for vad_index, vad_row in vad_df.iterrows():
        English_Word = vad_row["English Word"].strip()
        print(English_Word, flush=True)
        new_row = {}
        new_row["English Word"] = English_Word
        for column in vad_hebrew_columns:
            new_row[column] = vad_row[column]
        for emo_index, emo_row in emo_df.iterrows():
            if emo_row["English Word"].strip() == English_Word:
                for emotion in emotions_attributes:
                    new_row[emotion] = emo_row[emotion]
                merged_rows.append(new_row)
                break
    df = pd.DataFrame.from_records(merged_rows)
    df.to_csv(new_lexicon_output_path, index=False, encoding="utf-8")


def calc_attributes_statistics_per_chair(attribute,session_type, per_protocol_dict_name, per_protocol_graded_words_out_of_total_words_dict_name ):
    global now, current_time
    per_chair_dict_name = f'{attribute}_{session_type}_per_chair_frequency_dict'
    if CREATE_NEW_PER_CHAIR_RESULTS:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            f'started creating per chair grades for {attribute} for session {session_type}.  current time: {current_time}',
            flush=True)

        protocol_name_and_date_csv_file_path = os.path.join(processed_knesset_data_path, "knesset_data_csv_files", f"{session_type}_protocols_name_and_date.csv")
        calc_attribute_frequency_per_chair(protocol_name_and_date_csv_file_path, attribute, per_protocol_dict_name,
                                           per_protocol_graded_words_out_of_total_words_dict_name, per_chair_dict_name)
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(
            f'finished creating per chair grades for {attribute} for session {session_type}.   current time: {current_time}',
            flush=True)
    per_chair_dict = load_object(per_chair_dict_name)
    data = list(per_chair_dict.values())
    for key, val in per_chair_dict.items():
        print(f'{key}: {val}')
    apply_Mann_Kendall_Test(data, significance_level=0.05, data_name=per_chair_dict_name)

def create_per_protocol_session_name_dict(processed_protocols_path, output_dict_name):
    per_protocol_session_name_dict = {}
    protocol_names = os.listdir(processed_protocols_path)
    for protocol_name in protocol_names:
        protocol_path = os.path.join(processed_protocols_path, protocol_name)
        try:
            with open(protocol_path, encoding="utf-8") as file:
                protocol_json = json.load(file)
        except Exception as e:
            print(f'couldnt load json:{protocol_name} error was: {e}')
            continue
        session_name = protocol_json["session_name"]
        if "parent_session_name" in protocol_json:
            parent_session_name = protocol_json["parent_session_name"]
        else:
            print(f'in protocol {protocol_name} no parent_session_name')
            parent_session_name = None
        original_protocol_name = protocol_name.split(".jsonl")[0].strip()
        per_protocol_session_name_dict[original_protocol_name] = {"session_name": session_name, "parent_session_name": parent_session_name}
    save_object(per_protocol_session_name_dict, output_dict_name)



def calc_trend_in_session_attribute_csv(session_stats_path):
    df = pd.read_csv(session_stats_path)
    for col_name, column_values in df.items():
        if "fraction" in col_name:
            apply_Mann_Kendall_Test(list(column_values), significance_level=0.05, data_name=col_name)

def create_emotion_statistics_csv_for_committee_name(attribute, committtee_names_strings, per_protocol_dict, per_protocol_graded_words_out_of_total_words_dict, per_protocol_session_dict, csv_output_path):
    protocols_attribute_statistics = []
    for protocol in per_protocol_dict:
        is_session_protocol = False
        protocol_statistics_dict = {}
        protocol_session_dict = per_protocol_session_dict.get(protocol, None)
        if protocol_session_dict:
            protocol_session_name = protocol_session_dict["session_name"]
            if protocol_session_name:
                protocol_session_name = protocol_session_name.strip()
            protocol_parent_session_name = protocol_session_dict["parent_session_name"]
            if protocol_parent_session_name:
                protocol_parent_session_name = protocol_parent_session_name.strip()
            for name in committtee_names_strings:
                if protocol_session_name == name.strip() or protocol_parent_session_name == name.strip():
                    is_session_protocol = True
                    break
            if is_session_protocol:
                protocol_sum_of_scores, protocol_num_of_graded_tokens = per_protocol_dict.get(protocol, (-1, -1))
                if protocol_num_of_graded_tokens == -1:
                    print(f'protocol not in protocols dict: {protocol}')
                    continue
                protocol_num_of_lexicon_words, total_num_of_tokens_in_protocol = per_protocol_graded_words_out_of_total_words_dict.get(
                    protocol, (-1, -1))
                if total_num_of_tokens_in_protocol == -1:
                    print(f'protocol not in per_protocol_graded_words_out_of_total_words_dict: {protocol}')
                    continue
                if protocol_num_of_graded_tokens>0:
                    fraction_of_attribute_words_scores_out_lexicon_words_in_protocol = protocol_sum_of_scores/protocol_num_of_graded_tokens
                else:
                    print(f'protocol_num_of_graded_tokens is zero: {protocol}')
                    continue
                if total_num_of_tokens_in_protocol>0:
                    fraction_of_attribute_words_scores_out_of_all_words_in_protocol = protocol_sum_of_scores/total_num_of_tokens_in_protocol
                else:
                    print(f'total_num_of_tokens_in_protocol is zero in peotocol {protocol}')
                    continue
                protocol_statistics_dict["protocol_name"] = protocol
                protocol_statistics_dict[f"protocol_sum_of_scores_for_{attribute}"] = protocol_sum_of_scores
                protocol_statistics_dict["protocol_num_of_graded_tokens"] = protocol_num_of_graded_tokens
                protocol_statistics_dict[f"fraction_of_{attribute}_words_out_of_lexicon_words_in_protocol"] = fraction_of_attribute_words_scores_out_lexicon_words_in_protocol
                protocol_statistics_dict[f"fraction_of_{attribute}_words_out_of_all_words_in_protocol"] = fraction_of_attribute_words_scores_out_of_all_words_in_protocol
                # protocol_statistics_dict[f"fraction_of_lexicon_words_out_of_total_number_of_tokens"] = protocol_num_of_lexicon_words/total_num_of_tokens_in_protocol
                protocols_attribute_statistics.append(protocol_statistics_dict)
    df = pd.DataFrame.from_records(protocols_attribute_statistics)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    if not os.path.exists(os.path.dirname(csv_output_path)):
        os.makedirs(os.path.dirname(csv_output_path))
    df.to_csv(csv_output_path, index=False)


def calc_statistics_over_all_protocols_in_corpus():
    protocol_graded_words_out_of_total_words_dict = load_object(per_protocol_graded_words_out_of_total_words_dict_name)
    num_of_lexicon_words_in_all_protocols = 0
    num_of_tokens_in_all_protocols = 0
    for protocol in protocol_graded_words_out_of_total_words_dict:
        protocol_num_of_lexicon_words, protocol_total_num_of_tokens = protocol_graded_words_out_of_total_words_dict[
            protocol]
        num_of_lexicon_words_in_all_protocols += protocol_num_of_lexicon_words
        num_of_tokens_in_all_protocols += protocol_total_num_of_tokens
    print(f'num_of_lexicon_words_in_all_protocols: {num_of_lexicon_words_in_all_protocols}')
    print(f'num_of_tokens_in_all_protocols: {num_of_tokens_in_all_protocols}')
    num_of_lexicon_words_out_of_total_number_of_tokens = (
                                                                     num_of_lexicon_words_in_all_protocols / num_of_tokens_in_all_protocols) * 100
    print("num_of_lexicon_words_out_of_total_number_of_tokens:")
    print(f"{'%.2f' % num_of_lexicon_words_out_of_total_number_of_tokens}%")
    num_of_lexicon_words_in_all_protocols = 0
    sum_of_protocols_freqencies = 0
    for protocol in protocol_graded_words_out_of_total_words_dict:
        protocol_num_of_lexicon_words, protocol_total_num_of_tokens = protocol_graded_words_out_of_total_words_dict[
            protocol]
        if protocol_total_num_of_tokens != 0:
            protocol_graded_tokens_freq = protocol_num_of_lexicon_words / protocol_total_num_of_tokens
        else:
            continue
        sum_of_protocols_freqencies += protocol_graded_tokens_freq
    protocol_avg_lexicon_words_frequency = sum_of_protocols_freqencies / len(
        list(protocol_graded_words_out_of_total_words_dict.keys())) * 100
    print("per_protocol_avg_lexicon_words_frequency:")
    print(f"{'%.2f' % protocol_avg_lexicon_words_frequency}%")


if __name__ == '__main__':
    csv_emotion_lexicon_path = "emotions_hebrew_english_original_lexicon.csv"
    vad_lexicon_path = os.path.join(project_folder_path, "vad_manual_annotations_hebrew", "enriched_final_lexicon.csv")
    emotion_with_hebrew_annotations_path =   os.path.join(project_folder_path, "emotions", "emotion_lexicon_with_annotated_hebrew_words.csv")

    CREATE_EMO_WITH_HEBREW_LEXICON = False
    CREATE_NEW_GRADE_DICTS = True
    CREATE_PER_PROTOCOL_DICT = False
    CREATE_NEW_PER_CHAIR_RESULTS = False
    CALC_PER_CHAIR = False
    CREATE_PER_PROTOCOL_SESSION_NAME_DICT = False
    if CREATE_EMO_WITH_HEBREW_LEXICON:
        add_vad_hebrew_annotations_to_emo_lexicon(vad_lexicon_path, csv_emotion_lexicon_path, emotion_with_hebrew_annotations_path)
        print(f'finished creating emo lexicon', flush=True)

    if CREATE_PER_PROTOCOL_SESSION_NAME_DICT:
        processed_protocols_path = committee_processed_protocols_path
        per_protocol_session_name_dict_name = "per_protocol_session_name_dict"
        create_per_protocol_session_name_dict(processed_protocols_path, per_protocol_session_name_dict_name)


    emotion_attributes = ["anger","anticipation","disgust","fear","joy","negative","positive","sadness","surprise","trust"]
    vad_attributes = ["Valence", "Arousal", "Dominance"]
    session_types = ["committee", "plenary"]
    all_attributes = []
    all_attributes.extend(vad_attributes)
    all_attributes.extend(emotion_attributes)

    attribute = "trust"  #########################CHANGE ACCORDINGLY########################
    session_type = "committee"
    if CREATE_NEW_GRADE_DICTS:
        for attribute in all_attributes:
            if attribute in vad_attributes:
                lexicon_path = vad_lexicon_path
            else:
                lexicon_path = emotion_with_hebrew_annotations_path
            lexicon_grade_dict_name = f"{attribute}_word_grade_dict"

            now = datetime.now()
            current_time = now.strftime("%H:%M.:%S")
            print(f"started creating {attribute} grade dict. current time: {current_time}", flush=True)
            lexicon_grade_dict = calc_lexicon_word_attribute_grade_dict(attribute, lexicon_path, word_grade_dict_name=lexicon_grade_dict_name)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"finished creating {attribute} grade dict. current time: {current_time}", flush=True)
    else:
        lexicon_grade_dict_name = f"{attribute}_word_grade_dict"
        lexicon_grade_dict = load_object(lexicon_grade_dict_name)


    if attribute in emotion_attributes:
        attribute_type = "emotions"
    else:
        attribute_type = "vad"

    per_protocol_dict_name = f"{attribute}_{session_type}_grades_sum_and_count_per_protocol_dict"
    per_protocol_graded_words_out_of_total_words_dict_name=f"{session_type}_{attribute_type}_protocol_graded_words_out_of_total_words_dict_name"
    if CREATE_PER_PROTOCOL_DICT:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'started creating per protocol grades for {attribute} for session: {session_type}. current time: {current_time}', flush=True)

        if session_type == "plenary":
            processed_protocols_path = plenary_processed_protocols_path
        elif session_type == "committee":
            processed_protocols_path = committee_processed_protocols_path
        else:
            raise(f"wrong session type: {session_type}")

        calc_attribute_sum_of_grades_and_num_of_lexicon_words_per_protocol(processed_protocols_path,
                                                                           lexicon_grade_dict,
                                                                           output_dict_name=per_protocol_dict_name,
                                                                           per_protocol_graded_words_out_of_total_words_dict_name=per_protocol_graded_words_out_of_total_words_dict_name)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'finished creating per protocol grades for {attribute} for session: {session_type}. current time: {current_time}', flush=True)
    if CALC_PER_CHAIR:
        calc_attributes_statistics_per_chair(attribute,session_type, per_protocol_dict_name, per_protocol_graded_words_out_of_total_words_dict_name)

    #OPTION 1:
    # calc_statistics_over_all_protocols_in_corpus()
    #OPTION 2:
    session_name_strings = ["הוועדה לקידום מעמד האישה ולשוויון מגדרי", "הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה ולשוויון מגדרי" ]
    # session_name_strings = ["ועדת החוץ והביטחון", " ועדת החוץ והביטחון"]
    # session_name_strings = ["ועדת הכספים", " ועדת הכספים", " ועדת הכספים "]
    # session_name_strings = ["ועדת הכלכלה", " ועדת הכלכלה"]
    # session_name_strings = ["ועדת החוקה, חוק ומשפט", " ועדת החוקה חוק ומשפט"]
    # session_name_strings = ["ועדת החינוך, התרבות והספורט", "ועדת החינוך והתרבות"," ועדת החינוך, התרבות והספורט"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות", "ועדת הבריאות","ועדת העבודה והרווחה"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות","ועדת הבריאות"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות","ועדת העבודה והרווחה"]
    # session_name_strings = ["ועדת הפנים והגנת הסביבה","ועדת הפנים ואיכות הסביבה", " ועדת הפנים והגנת הסביבה"]
    # session_name_strings = ["הוועדה לענייני ביקורת המדינה"," הוועדה לענייני ביקורת המדינה"]
    # session_name_strings = ["ועדת העלייה, הקליטה והתפוצות"]
    # session_name_strings = ["ועדת הכנסת", " ועדת הכנסת"]
    # session_name_strings = ["הוועדה המיוחדת לזכויות הילד", "הוועדה המיוחדת לקידום מעמד הילד"]
    # session_name_strings = ["הוועדה המיוחדת לפניות הציבור"]
    # session_name_strings = ["ועדת המדע והטכנולוגיה", "הוועדה המיוחדת לענייני מחקר ופיתוח מדעי וטכנולוגי"]
    # session_name_strings = ["הוועדה המיוחדת לבחינת בעיית העובדים הזרים", "הוועדה המיוחדת לעובדים זרים"]
    # session_name_strings = ["הוועדה המיוחדת ליישום הנגשת המידע הממשלתי ועקרונות שקיפותו לציבור"]
    # session_name_strings = ["הוועדה המיוחדת למאבק בנגעי הסמים והאלכוהול", "הוועדה המיוחדת למאבק בנגע הסמים", "הוועדה המיוחדת לענייני התמכרויות, סמים ואתגרי הצעירים בישראל", "הוועדה המיוחדת להתמודדות עם סמים ואלכוהול"]
    # session_name_strings = ["ועדת המשנה לתקנות שוויון זכויות לאנשים עם מוגבלות", "ועדת המשנה ליישום חוק שוויון זכויות לאנשים עם מוגבלות"]
    # session_name_strings = ["ועדת ביטחון הפנים"]
    # session_name_strings = ["הוועדה המיוחדת לעניין נגיף הקורונה החדש ולבחינת היערכות המדינה למגפות ולרעידות אדמה", "הוועדה המיוחדת בעניין ההתמודדות עם נגיף הקורונה - זמנית"]
    # session_name_strings = ["הוועדה המיוחדת לצדק חלוקתי ולשוויון חברתי"]
    # session_name_strings = ["הוועדה המסדרת"]
    all_sessions_names_strings = [
        ["הוועדה לקידום מעמד האישה ולשוויון מגדרי", "הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה ולשוויון מגדרי"]
        , ["ועדת החוץ והביטחון", " ועדת החוץ והביטחון"]
        , ["ועדת הכספים", " ועדת הכספים", " ועדת הכספים "]
        , ["ועדת הכלכלה", " ועדת הכלכלה"]
        , ["ועדת החוקה, חוק ומשפט", " ועדת החוקה חוק ומשפט"]
        , ["ועדת החינוך, התרבות והספורט", "ועדת החינוך והתרבות", " ועדת החינוך, התרבות והספורט"]
        , ["ועדת העבודה, הרווחה והבריאות", "ועדת הבריאות", "ועדת העבודה והרווחה"]
        , ["ועדת העבודה, הרווחה והבריאות", "ועדת הבריאות"]
        , ["ועדת העבודה, הרווחה והבריאות", "ועדת העבודה והרווחה"]
        , ["ועדת הפנים והגנת הסביבה", "ועדת הפנים ואיכות הסביבה", " ועדת הפנים והגנת הסביבה"]
        , ["הוועדה לענייני ביקורת המדינה", " הוועדה לענייני ביקורת המדינה"]
        , ["ועדת העלייה, הקליטה והתפוצות"]
        , ["ועדת הכנסת", " ועדת הכנסת"]
        , ["הוועדה המיוחדת לזכויות הילד", "הוועדה המיוחדת לקידום מעמד הילד"]
        , ["הוועדה המיוחדת לפניות הציבור"]
        , ["ועדת המדע והטכנולוגיה", "הוועדה המיוחדת לענייני מחקר ופיתוח מדעי וטכנולוגי"]
        , ["הוועדה המיוחדת לבחינת בעיית העובדים הזרים", "הוועדה המיוחדת לעובדים זרים"]
        , ["הוועדה המיוחדת ליישום הנגשת המידע הממשלתי ועקרונות שקיפותו לציבור"]
        , ["הוועדה המיוחדת למאבק בנגעי הסמים והאלכוהול", "הוועדה המיוחדת למאבק בנגע הסמים", "הוועדה המיוחדת לענייני התמכרויות, סמים ואתגרי הצעירים בישראל", "הוועדה המיוחדת להתמודדות עם סמים ואלכוהול"]
        , ["ועדת המשנה לתקנות שוויון זכויות לאנשים עם מוגבלות", "ועדת המשנה ליישום חוק שוויון זכויות לאנשים עם מוגבלות"]
        , ["ועדת ביטחון הפנים"]
        , ["הוועדה המיוחדת לעניין נגיף הקורונה החדש ולבחינת היערכות המדינה למגפות ולרעידות אדמה", "הוועדה המיוחדת בעניין ההתמודדות עם נגיף הקורונה - זמנית"]
        , ["הוועדה המיוחדת לצדק חלוקתי ולשוויון חברתי"]
        , ["הוועדה המסדרת"]]

    session_english_name = "women_status"
    # session_english_name = "security_and_out"
    # session_english_name = "funds"
    # session_english_name = "economy"
    # session_english_name = "constitution"
    # session_english_name = "education_culture_sports"
    # session_english_name = "work_welfare_health"
    # session_english_name = "work_health"
    # session_english_name = "work_welfare"
    # session_english_name = "in_and_environment"
    # session_english_name = "state_critisism"
    # session_english_name = "aliya"
    # session_english_name = "knesset"
    # session_english_name = "child_rights"
    # session_english_name = "public_rights"
    # session_english_name = "technology"
    # session_english_name = "foreign_workers"
    # session_english_name = "access_to_information"
    # session_english_name = "people_with_disabilities"
    # session_english_name = "inner_securitiy"
    # session_english_name = "covid"
    # session_english_name = "justice_social_equality"
    # session_english_name = "organizing"
    all_sessions_english_names = ["women_status"
        , "security_and_out"
        , "funds"
        , "economy"
        , "constitution"
        , "education_culture_sports"
        , "work_welfare_health"
        , "work_health"
        , "work_welfare"
        , "in_and_environment"
        , "state_critisism"
        , "aliya"
        , "knesset"
        , "child_rights"
        , "public_rights"
        , "technology"
        , "foreihn_workers"
        , "access_to_information"
        , "people_with_disabilities"
        , "inner_securitiy"
        , "covid"
        , "justice_social_equality"
        , "organizing"]

    per_protocol_dict = load_object(per_protocol_dict_name)
    per_protocol_graded_words_out_of_total_words_dict = load_object(
        per_protocol_graded_words_out_of_total_words_dict_name)
    per_protocol_session_dict = load_object("per_protocol_session_name_dict")
    outputs_path = os.path.join(project_folder_path, "emotions")
    all_committees_together = True
    if all_committees_together:

        output_trends_text_file_path = os.path.join(outputs_path, f'{attribute}-all-committees_counting-trends.txt')
        if os.path.exists(output_trends_text_file_path):
            os.remove(output_trends_text_file_path)

        for session_name_strings, session_english_name in zip(all_sessions_names_strings, all_sessions_english_names):
            print(session_english_name)
            with open(output_trends_text_file_path,'a') as output_file:
                output_file.write(f'{session_english_name}:\n')
            csv_output_name = f'{attribute}_{session_english_name}_stats.csv'
            csv_output_path = os.path.join(outputs_path,"per_session_stats", session_english_name, csv_output_name)

            create_emotion_statistics_csv_for_committee_name(attribute, session_name_strings, per_protocol_dict,
                                                             per_protocol_graded_words_out_of_total_words_dict,
                                                             per_protocol_session_dict, csv_output_path)
            with open(output_trends_text_file_path, 'a') as output_file:
                with redirect_stdout(output_file):
                    calc_trend_in_session_attribute_csv(csv_output_path)
                    print('\n')

    else:
        csv_output_name = f'{attribute}_{session_english_name}_stats.csv'
        csv_output_path = os.path.join(outputs_path,"per_session_stats",session_english_name, csv_output_name)
        create_emotion_statistics_csv_for_committee_name(attribute, session_name_strings, per_protocol_dict,per_protocol_graded_words_out_of_total_words_dict, per_protocol_session_dict , csv_output_path)
        calc_trend_in_session_attribute_csv(csv_output_path)
