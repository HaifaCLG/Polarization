import os

import pandas as pd

from aux_functions import parse_each_conllu_sentence_separatly
from params_and_config import *

israel_path = os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "israel_vad.csv" )
avia_path = os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "avia_vad.csv" )
shira_path = os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "shira_vad.csv" )


def get_annotator_hebrew_word(row):
    hebrew_word = row["Hebrew Undotted"]
    if not hebrew_word:
        hebrew_word = row["Hebrew Word"]
    if row["NOT VALID"] =="V" or row["NOT VALID"] == "v":
        hebrew_word = "not_valid"
    return hebrew_word


def check_alternative_agreement(israel_row, avia_row, shira_row):
    israel_alternatives = get_all_annotators_alternatives(israel_row)
    avia_alternatives = get_all_annotators_alternatives(avia_row)
    shira_alternatives = get_all_annotators_alternatives(shira_row)
    for word in israel_alternatives:
        if word in avia_alternatives and word in shira_alternatives:
            return True
    return False


def get_all_annotators_alternatives(row):
    alternatives = []
    hebrew_word = get_annotator_hebrew_word(row)
    alternatives.append(hebrew_word)
    israel_altervative_1 = row["Alternative 1"]
    if israel_altervative_1:
        alternatives.append(israel_altervative_1)
    israel_altervative_2 = row["Alternative 2"]
    if israel_altervative_2:
        alternatives.append(israel_altervative_2)
    return alternatives
def check_lemma_agreement(israel_row, avia_row, shira_row):
    if israel_row["Hebrew Lemma"] == avia_row["Hebrew Lemma"] == shira_row["Hebrew Lemma"]:
        return True
    else:
        print(
            f"no LEMMA agreement on word {'english_word'}: israel: {israel_row['Hebrew Lemma']} | avia: {avia_row['Hebrew Lemma']} | shira: {shira_row['Hebrew Lemma']}")
        return False

def merge_annotations_of_shared_lines(israel_file, shira_file, avia_file, israel_max_line, shira_max_line, avia_max_line, output_lexicon):
    israel_df = pd.read_csv(israel_file, keep_default_na=False)
    shira_df = pd.read_csv(shira_file, keep_default_na=False)
    avia_df = pd.read_csv(avia_file, keep_default_na=False)
    new_lines_list = []
    max_num_of_alternatives = 7
    for (index, israel_line), (_,shira_line), (_,avia_line) in zip(israel_df.iterrows(), shira_df.iterrows(), avia_df.iterrows()):
        if index >avia_max_line:
            break
        if not (israel_line["English Word"].lower() == shira_line["English Word"].lower() == str(avia_line["English Word"]).lower()):
            print(f"error not same word in line. lines are not aligned! israel: {israel_line['English Word']}, shira: {shira_line['English Word']}, avia: {avia_line['English Word']}")
            return
        line = {}
        line["English Word"] = avia_line["English Word"].strip()
        line["Valence"] = avia_line["Valence"]
        line["Arousal"] = avia_line["Arousal"]
        line["Dominance"] = avia_line["Dominance"]
        line["Hebrew Word"] = avia_line["Hebrew Word"].strip()
        line["Hebrew Undotted"] = avia_line["Hebrew Undotted"].strip()#avia max lines is the highest
        line["Hebrew Lemma"] = avia_line["Hebrew Lemma"].strip()
        alternatives = set()
        alternatives.add(line["Hebrew Undotted"])
        alternatives.add(line["Hebrew Lemma"])
        if avia_line["Alternative 1"]:
            alternatives.add(avia_line["Alternative 1"].strip())
        if avia_line["Alternative 2"]:
            alternatives.add(avia_line["Alternative 2"].strip())
        if index <= israel_max_line - 1:
            if israel_line["Hebrew Undotted"]:
                alternatives.add(israel_line["Hebrew Undotted"].strip())
            if israel_line["Hebrew Lemma"]:
                alternatives.add(israel_line["Hebrew Lemma"].strip())
            if israel_line["Alternative 1"]:
                alternatives.add(israel_line["Alternative 1"].strip())
            if israel_line["Alternative 2"]:
                alternatives.add(israel_line["Alternative 2"].strip())
        if index <= shira_max_line - 1:
            if shira_line["Hebrew Undotted"]:
                alternatives.add(shira_line["Hebrew Undotted"].strip())
            if shira_line["Hebrew Lemma"]:
                alternatives.add(shira_line["Hebrew Lemma"].strip())
            if shira_line["Alternative 1"]:
                alternatives.add(shira_line["Alternative 1"].strip())
            if shira_line["Alternative 2"]:
                alternatives.add(shira_line["Alternative 2"].strip())
        if line["Hebrew Undotted"] in alternatives:
            alternatives.remove(line["Hebrew Undotted"])
        if line["Hebrew Lemma"] in alternatives:
            alternatives.remove(line["Hebrew Lemma"])
        num_of_alternatives = len(alternatives)
        # if num_of_alternatives > max_num_of_alternatives:
        #     max_num_of_alternatives = num_of_alternatives
        alternative_num = 1
        for val in alternatives:
            line[f"Alternative {alternative_num}"] = val
            alternative_num +=1
        line["NOT VALID"] = avia_line["NOT VALID"].strip()
        new_lines_list.append(line)
    columns= ["English Word","Valence", "Arousal", "Dominance", "Hebrew Word", "Hebrew Undotted",  "Hebrew Lemma"]
    first_time = True
    for line in new_lines_list:
        if f"Alternative {max_num_of_alternatives}" not in line:
            not_valid = line["NOT VALID"]
            line.pop("NOT VALID")
            for num in range(1, max_num_of_alternatives+1):
                if first_time:
                    columns.append(f"Alternative {num}")
                if f"Alternative {num}" in line:
                    continue
                else:
                    line[f"Alternative {num}"] = ""
            first_time = False
            line["NOT VALID"] = not_valid
    columns.append("NOT VALID")
    df = pd.DataFrame.from_records(new_lines_list)
    df.to_csv(output_lexicon, index=False, columns=columns)

def merge_all_lexicons(path_to_lexicon_files, output_path):
    names = os.listdir(path_to_lexicon_files)
    max_num_of_alternatives = 7
    all_lexicon_dfs = []
    for name in names:
        if ".csv" in name and "final" not in name:
            lexicon_df = pd.read_csv(os.path.join(path_to_lexicon_files, name), keep_default_na=False)
            df_colls = list(lexicon_df.columns.values)
            if "NOT VALID" not in df_colls:
                lexicon_df["NOT VALID"] = " "
            if f"Alternative {max_num_of_alternatives}" not in df_colls:
                for num in range(1, max_num_of_alternatives + 1):
                    if f"Alternative {num}" in df_colls:
                        continue
                    else:
                        lexicon_df.insert(loc=len(lexicon_df.columns) - 1, column=f"Alternative {num}", value='')
            all_lexicon_dfs.append(lexicon_df)
    concatenated_df = pd.concat(all_lexicon_dfs, ignore_index=True)
    concatenated_df['English Word'] = concatenated_df['English Word'].str.lower()

    sorted_df = concatenated_df.sort_values(by=["English Word"])
    sorted_df.to_csv(output_path, index=False)






def check_agreement_for_vad_annotations():
    israel_annotations_df = pd.read_csv(israel_path, keep_default_na=False)
    avia_annotations_df = pd.read_csv(avia_path, keep_default_na=False)
    shira_annotations_df = pd.read_csv(shira_path, keep_default_na=False)
    num_of_rows_to_compare = 392
    num_of_agreements = 0
    num_of_alt_agreements = 0
    num_of_lemma_agreements = 0
    for (idxRow, israel_row), (_, avia_row), (_, shira_row) in zip(israel_annotations_df.iterrows(),
                                                                   avia_annotations_df.iterrows(),
                                                                   shira_annotations_df.iterrows()):
        if idxRow + 1 == num_of_rows_to_compare:
            break
        lemma_res = check_lemma_agreement(israel_row, avia_row, shira_row)
        if lemma_res:
            num_of_lemma_agreements += 1
        english_word = israel_row["English Word"]
        assert (english_word == avia_row["English Word"])
        assert (english_word == shira_row["English Word"])
        israel_hebrew_word = get_annotator_hebrew_word(israel_row)
        avia_hebrew_word = get_annotator_hebrew_word(avia_row)
        shira_hebrew_word = get_annotator_hebrew_word(shira_row)

        if israel_hebrew_word == avia_hebrew_word == shira_hebrew_word:
            num_of_agreements += 1
        elif check_alternative_agreement(israel_row, avia_row, shira_row):
            num_of_alt_agreements += 1
            print(f"altervative agreement on word: {english_word}")
        else:
            pass
            print(
                f"no agreement on word {english_word}: israel: {israel_hebrew_word} | avia: {avia_hebrew_word} | shira: {shira_hebrew_word}")
    print(
        f"agreement is: {num_of_agreements}/{num_of_rows_to_compare} which is: {'%.2f' % ((num_of_agreements / num_of_rows_to_compare) * 100)}%")
    print(
        f"alternative agreement is: {(num_of_agreements + num_of_alt_agreements)}/{num_of_rows_to_compare} which is: {'%.2f' % (((num_of_agreements + num_of_alt_agreements) / num_of_rows_to_compare) * 100)}%")
    print(
        f"lemma agreement is: {(num_of_lemma_agreements)}/{num_of_rows_to_compare} which is: {'%.2f' % (((num_of_lemma_agreements) / num_of_rows_to_compare) * 100)}%")


def extract_new_lemma_from_conllu_file(conllu_words, original_lemma):
    new_lemma = ""
    for conllu_word in conllu_words:
        word = conllu_word.metadata["text"].replace("הוא ביקש", "").strip()
        if original_lemma == word:
            if len(conllu_word) == 3:
                if conllu_word[2]['xpos'] == "VERB":
                    new_lemma = conllu_word[2]["lemma"]
                    break
            elif len(conllu_word) == 4:
                if conllu_word[3]['xpos'] == "VERB":
                    new_lemma = conllu_word[3]["lemma"]
                    break
    return new_lemma

def enrich_lexicon_csv_with_fixed_lemmas(lexicon_csv_path, lemma_conllu_file=None, lemmas_to_fix_file=None, output_lexicon="enriched_lexicon.csv"):
    lexicon_df = pd.read_csv(lexicon_csv_path, keep_default_na=False)
    lexicon_df = lexicon_df.dropna(how='any')
    alternative_hebrew_columns = ["Hebrew Lemma", "Alternative 1", "Alternative 2", "Alternative 3", "Alternative 4", "Alternative 5", "Alternative 6", "Alternative 7"]
    if lemma_conllu_file:
        with open(lemma_conllu_file, encoding="utf-8") as file:
            data = file.read()
        conllu_words = parse_each_conllu_sentence_separatly(data)
    if lemmas_to_fix_file:
        lemmas_to_fix_df = pd.read_csv(lemmas_to_fix_file )
    for index, row in lexicon_df.iterrows():
        new_hebrew_lemma = ""
        original_hebrew_lemma = ""
        alternatives = set()
        for column in alternative_hebrew_columns:
            if row[column].strip():
                word = row[column].strip()
                if column == "Hebrew Lemma":
                    original_hebrew_lemma = word
                alternatives.add(word)
                if lemmas_to_fix_file and word in list(lemmas_to_fix_df["original word"].values):
                    lemma_row = lemmas_to_fix_df.loc[lemmas_to_fix_df['original word'] == word]
                    new_word = lemma_row["fixed lemma"].values[0]
                else:
                    if lemma_conllu_file:
                        new_word = extract_new_lemma_from_conllu_file(conllu_words, word)
                if new_word:
                    alternatives.add(new_word)
                    if column == "Hebrew Lemma":
                        new_hebrew_lemma = new_word
        num_of_alternatives = len(alternatives)
        if num_of_alternatives>7:
            print(f"a too many alternatives. num of alternatives is: {num_of_alternatives}")
        if new_hebrew_lemma:
            lexicon_df.at[index,"Hebrew Lemma"] = new_hebrew_lemma
            alternatives.remove(new_hebrew_lemma)
        else:
            lexicon_df.at[index, "Hebrew Lemma"] = original_hebrew_lemma
            if original_hebrew_lemma in alternatives:
                alternatives.remove(original_hebrew_lemma)
        for word, i in zip(alternatives, range(1, num_of_alternatives+1)):
            lexicon_df.at[index, f'Alternative {i}'] = word

    lexicon_df.to_csv(output_lexicon,index=False)

if __name__ == '__main__':
    # check_agreement_for_vad_annotations()
    israel_first_lines_file=os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "israel_vad.csv")
    shira_first_lines_file=os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "shira_vad.csv")
    avia_first_lines_file=os.path.join(vad_manual_annotations_path, "first_400_annotations_for_comparison", "avia_vad.csv")
    merged_shared_lines_lexicon_path = os.path.join(vad_manual_annotations_path, "merged_shared_lines_lexicon_1-1999.csv")
    # merge_annotations_of_shared_lines(israel_file=israel_first_lines_file, shira_file=shira_first_lines_file, avia_file=avia_first_lines_file,israel_max_line=1175, shira_max_line=392, avia_max_line=1999, output_lexicon=merged_shared_lines_lexicon_path)
    # merge_all_lexicons(vad_manual_annotations_path, output_path=os.path.join(vad_manual_annotations_path, "almost_final_lexicon.csv")),

    enrich_lexicon_csv_with_fixed_lemmas(os.path.join(vad_manual_annotations_path, "almost_final_lexicon.csv"), lemma_conllu_file=lemma_conllu_file_path, lemmas_to_fix_file=os.path.join(vad_manual_annotations_path,"lemmas_to_fix","lemmas_to_fix.csv"),output_lexicon=enriched_final_lexicon_path)

