import json
import os
import shutil
from operator import itemgetter
from statistics import median, variance

import pandas
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from params_and_config import *
from statistic_functions import apply_Mann_Kendall_Test, perform_t_test
from vad_functions import *
from vad_models_predictions import predict_one_vad_value_on_sentences
from sentence_transformers import SentenceTransformer, models
from aux_functions import *


def get_shard_session_sentences(shard_path, session_name_strings):
    shard_relevant_sentences = []
    line_counter = 0
    try:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line_counter += 1
                try:
                    sent = json.loads(line)
                except Exception as e:
                    print(f'Could not load JSON in file {shard_path}. Line was: {line_counter}')
                    continue
                sent_session_name = str(sent["session_name"]).strip()
                sent_parent_session_name = str(sent["parent_session_name"]).strip()
                for session_name in session_name_strings:
                    if sent_session_name == str(session_name).strip() or sent_parent_session_name == str(session_name).strip():
                        shard_relevant_sentences.append(sent)
                        break

    except Exception as e:
        print(f"Could not read file {shard_path}. Error was: {e}")
        return []
    return shard_relevant_sentences


def add_vad_values_to_sentences_jsons(shard_relevant_sentences, v_predictions, a_predictions, d_predictions):
    sentences_with_vad = []
    for sent, v, a, d in zip(shard_relevant_sentences, v_predictions, a_predictions, d_predictions):
        sent["vad_values"] = {"valence": v, "arousal": a, "dominance": d}
        sentences_with_vad.append(sent)
    return sentences_with_vad


def predict_vad_values_on_sentences(sentences, sent_embeddings_model, tokenizer):
    v_predictions = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer,
                                                       vad_column="V", print=False)
    a_predictions = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer,
                                                       vad_column="A", print=False)
    d_predictions = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer,
                                                       vad_column="D", print=False)
    return v_predictions, a_predictions, d_predictions


def save_output_shard(shard_data, output_dir_path, shard_number):
    output_file_path = os.path.join(output_dir_path, f"vad_shard_{shard_number}.jsonl")

    with open(output_file_path, 'w', encoding="utf-8") as f:
        for sentence in shard_data:
            f.write(json.dumps(sentence, ensure_ascii=False) + '\n')

def temp_copy_lines_from_to_fix_file_that_are_women_committee(to_fix_file_path,session_name_strings,  output_file):
    wanted_names = [f'session_name": "{name.strip()}' for name in session_name_strings]
    with open(to_fix_file_path, encoding="utf-8") as file:
        for line in file:
            if any(name in line for name in wanted_names):
                with open(output_file, "a", encoding="utf-8") as out_file:
                    out_file.write(line)


def temp_replace_sentences_in_vad_shards(fixed_sentences_file, vad_shards_dir):
    df = pandas.read_json(fixed_sentences_file, lines=True, encoding="utf-8")
    protocol_names = list(set(list(df["protocol_name"])))
    shard_files = os.listdir(vad_shards_dir)
    for protocol_name in protocol_names:
        protocol_fixed_lines = df[df["protocol_name"]== protocol_name]
        for shard_name in shard_files:
            shard_lines = []
            shard_path = os.path.join(vad_shards_dir, shard_name)
            with open(shard_path, encoding="utf-8") as vad_file:
                vad_file_text = vad_file.read()
            if protocol_name in vad_file_text:
                with open(shard_path, encoding="utf-8") as vad_file:
                    for line in vad_file:
                        if protocol_name in line:
                            json_vad_line = json.loads(line)
                            vad_sentence_id = json_vad_line["sentence_id"]
                            vad_values = json_vad_line["vad_values"]
                            for i, row in protocol_fixed_lines.iterrows():
                                if row["_id"] == vad_sentence_id:
                                    new_line = dict(row)
                                    new_line["vad_values"] = vad_values
                                    shard_lines.append(json.dumps(new_line, ensure_ascii=False))
                                    break
                        else:
                            shard_lines.append(line.strip())
                if shard_lines:
                    temp_path = os.path.join(vad_shards_dir, f'temp_{shard_name}')
                    with open(temp_path, "w", encoding="utf-8") as file:
                        file.write('\n'.join(shard_lines) + '\n')

                    os.remove(shard_path)
                    os.rename(temp_path, shard_path)



def predict_vad_values_of_session_sentences(session_name_strings, input_dir_path, output_dir_path):
    output_shard_size = 10000
    chunk_size = 5000
    sent_embeddings_model = SentenceTransformer(SENT_TRANSFORMER_TYPE)
    tokenizer = None
    output_shard_relevant_sentences = []
    output_shard_number = 0
    shards = os.listdir(input_dir_path)

    for shard in shards:
        shard_num = int(shard.split("shard_")[1].split(".jsonl")[0])
        print(shard_num)
        print(f'length of output_shard_relevant_sentences: {len(output_shard_relevant_sentences)}')
        shard_path = os.path.join(input_dir_path, shard)
        shard_relevant_sentences = get_shard_session_sentences(shard_path, session_name_strings)
        if not shard_relevant_sentences:
            print(f'No session sentences in shard: {shard_path}')
            continue
        sentences_texts = [sent["sentence_text"] for sent in shard_relevant_sentences]
        sentences = process_input_texts_for_multi_model(sentences_texts)
        chunk_counter = 0
        for i in range(0, len(sentences), chunk_size):
            print(f'chunk:{chunk_counter}')
            chunk_counter+=1
            chunk = sentences[i:i + chunk_size]
            chunk_relevant_sentences = shard_relevant_sentences[i:i + chunk_size]

            v_predictions, a_predictions, d_predictions = predict_vad_values_on_sentences(chunk, sent_embeddings_model,
                                                                                          tokenizer)
            chunk_sentences_with_vad = add_vad_values_to_sentences_jsons(chunk_relevant_sentences, v_predictions,
                                                                         a_predictions, d_predictions)
            # chunk_sentences_with_vad_as_strings = [json.dumps(sent).strip() for sent in chunk_sentences_with_vad]

            if len(output_shard_relevant_sentences) + len(chunk_sentences_with_vad) < output_shard_size:
                output_shard_relevant_sentences.extend(chunk_sentences_with_vad)
            else:
                for line in chunk_sentences_with_vad:
                    if len(output_shard_relevant_sentences) < output_shard_size:
                        output_shard_relevant_sentences.append(line)
                    else:
                        print(f'saving session_shard {output_shard_number}')
                        save_output_shard(output_shard_relevant_sentences, output_dir_path, output_shard_number)
                        output_shard_relevant_sentences = [line]
                        output_shard_number += 1

    if output_shard_relevant_sentences:
        save_output_shard(output_shard_relevant_sentences, output_dir_path, output_shard_number)


def create_vad_avg_and_median_per_coalition_opposition_csvs(vad_shards_dir, coalition_output_path, opposition_output_path, filter_short_sentences=False,
                                               filter_non_hebrew=False):
    vad_shards = os.listdir(vad_shards_dir)
    coalition_vad_values = {}
    opposition_vad_values = {}
    for vad_shard in vad_shards:
        vad_shard_path = os.path.join(vad_shards_dir, vad_shard)
        with open(vad_shard_path, encoding="utf-8") as vad_file:
            for line in vad_file:
                sent_entity = json.loads(line)
                sent_text = sent_entity["sentence_text"]
                if filter_short_sentences and len(sent_text.split()) < 2:
                    continue
                if filter_non_hebrew and not is_hebrew(sent_text, print_non_hebrew=False):
                    continue
                protocol_name = sent_entity["protocol_name"]
                sent_v = sent_entity["vad_values"]["valence"]
                sent_a = sent_entity["vad_values"]["arousal"]
                sent_d = sent_entity["vad_values"]["dominance"]
                if "member_of_coalition_or_opposition" in sent_entity:
                    member_of_coalition_or_opposition = sent_entity["member_of_coalition_or_opposition"]
                    if member_of_coalition_or_opposition == "coalition":
                        coalition_protocol_values_dict = coalition_vad_values.get(protocol_name, None)
                        if not coalition_protocol_values_dict:
                            v_protocol_values = []
                            a_protocol_values = []
                            d_protocol_values = []
                            coalition_vad_values[protocol_name] = {"v_values": v_protocol_values,
                                                                   "a_values": a_protocol_values,
                                                                   "d_values": d_protocol_values}
                            coalition_protocol_values_dict = coalition_vad_values[protocol_name]
                        coalition_protocol_values_dict["v_values"].append(sent_v)
                        coalition_protocol_values_dict["a_values"].append(sent_a)
                        coalition_protocol_values_dict["d_values"].append(sent_d)
                    elif member_of_coalition_or_opposition == "opposition":
                        opposition_protocol_values_dict = opposition_vad_values.get(protocol_name, None)
                        if not opposition_protocol_values_dict:
                            v_protocol_values = []
                            a_protocol_values = []
                            d_protocol_values = []
                            opposition_vad_values[protocol_name] = {"v_values": v_protocol_values,
                                                                   "a_values": a_protocol_values,
                                                                   "d_values": d_protocol_values}
                            opposition_protocol_values_dict = opposition_vad_values[protocol_name]
                        opposition_protocol_values_dict["v_values"].append(sent_v)
                        opposition_protocol_values_dict["a_values"].append(sent_a)
                        opposition_protocol_values_dict["d_values"].append(sent_d)
                    else:
                        continue
                else:
                    continue

    coalition_protocols_vad_statistics = calculate_protocols_vad_statistics(coalition_vad_values)
    opposition_protocols_vad_statistics = calculate_protocols_vad_statistics(opposition_vad_values)
    df = pd.DataFrame.from_records(coalition_protocols_vad_statistics)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    df.to_csv(coalition_output_path, index=False)

    df = pd.DataFrame.from_records(opposition_protocols_vad_statistics)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    df.to_csv(opposition_output_path, index=False)
def create_vad_avg_and_median_per_protocol_csv(vad_shards_dir, output_path, filter_short_sentences=False, filter_non_hebrew=False):
    vad_shards = os.listdir(vad_shards_dir)
    protocols_vad_values = {}
    for vad_shard in vad_shards:
        vad_shard_path = os.path.join(vad_shards_dir, vad_shard)
        with open(vad_shard_path, encoding="utf-8") as vad_file:
            for line in vad_file:
                sent_entity = json.loads(line)
                sent_text = sent_entity["sentence_text"]
                if filter_short_sentences and len(sent_text.split()) < 2:
                    continue
                if filter_non_hebrew and not is_hebrew(sent_text, print_non_hebrew=False):
                    continue
                protocol_name = sent_entity["protocol_name"]
                sent_v = sent_entity["vad_values"]["valence"]
                sent_a = sent_entity["vad_values"]["arousal"]
                sent_d = sent_entity["vad_values"]["dominance"]

                protocol_values_dict = protocols_vad_values.get(protocol_name, None)
                if not protocol_values_dict:
                    v_protocol_values = []
                    a_protocol_values = []
                    d_protocol_values = []
                    protocols_vad_values[protocol_name] = {"v_values": v_protocol_values, "a_values": a_protocol_values, "d_values": d_protocol_values}
                    protocol_values_dict = protocols_vad_values[protocol_name]
                protocol_values_dict["v_values"].append(sent_v)
                protocol_values_dict["a_values"].append(sent_a)
                protocol_values_dict["d_values"].append(sent_d)

    protocols_vad_statistics = calculate_protocols_vad_statistics( protocols_vad_values)
    df = pd.DataFrame.from_records(protocols_vad_statistics)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    df.to_csv(output_path, index=False)


def calculate_protocols_vad_statistics(protocols_vad_values):
    protocols_vad_statistics = []
    for protocol, vad_dict in protocols_vad_values.items():
        protocol_statistics_dict = {}
        protocol_statistics_dict["protocol_name"] = protocol
        protocol_statistics_dict["protocol_v_avg"] = sum(vad_dict["v_values"]) / len(vad_dict["v_values"])
        protocol_statistics_dict["protocol_v_median"] = median(vad_dict["v_values"])
        if len(vad_dict["v_values"]) < 2:
            print(f'not enough v values in protocol: {protocol}')
            protocol_statistics_dict["protocol_v_var"] = 0
        else:
            protocol_statistics_dict["protocol_v_var"] = variance(vad_dict["v_values"])
        protocol_statistics_dict["protocol_a_avg"] = sum(vad_dict["a_values"]) / len(vad_dict["a_values"])
        protocol_statistics_dict["protocol_a_median"] = median(vad_dict["a_values"])
        protocol_statistics_dict["protocol_d_avg"] = sum(vad_dict["d_values"]) / len(vad_dict["d_values"])
        protocol_statistics_dict["protocol_d_median"] = median(vad_dict["d_values"])

        assert (len(vad_dict["v_values"]) == len(vad_dict["a_values"]) == len(vad_dict["d_values"]))
        protocol_statistics_dict["num_of_sentences_in_protocol"] = len(vad_dict["v_values"])
        protocol_statistics_dict["num_of_high_v_sentences"] = len(
            [value for value in vad_dict["v_values"] if value > 0.7])
        protocol_statistics_dict["fraction_of_high_v_sentences"] = protocol_statistics_dict["num_of_high_v_sentences"] / \
                                                                   protocol_statistics_dict[
                                                                       "num_of_sentences_in_protocol"]

        protocol_statistics_dict["num_of_very_high_v_sentences"] = len(
            [value for value in vad_dict["v_values"] if value > 0.9])
        protocol_statistics_dict["fraction_of_very_high_v_sentences"] = protocol_statistics_dict[
                                                                            "num_of_very_high_v_sentences"] / \
                                                                        protocol_statistics_dict[
                                                                            "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_low_v_sentences"] = len(
            [value for value in vad_dict["v_values"] if value < 0.3])
        protocol_statistics_dict["fraction_of_low_v_sentences"] = protocol_statistics_dict["num_of_low_v_sentences"] / \
                                                                  protocol_statistics_dict[
                                                                      "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_very_low_v_sentences"] = len(
            [value for value in vad_dict["v_values"] if value < 0.1])
        protocol_statistics_dict["fraction_of_very_low_v_sentences"] = protocol_statistics_dict[
                                                                           "num_of_very_low_v_sentences"] / \
                                                                       protocol_statistics_dict[
                                                                           "num_of_sentences_in_protocol"]

        protocol_statistics_dict["num_of_high_a_sentences"] = len(
            [value for value in vad_dict["a_values"] if value > 0.7])
        protocol_statistics_dict["fraction_of_high_a_sentences"] = protocol_statistics_dict["num_of_high_a_sentences"] / \
                                                                   protocol_statistics_dict[
                                                                       "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_very_high_a_sentences"] = len(
            [value for value in vad_dict["a_values"] if value > 0.9])
        protocol_statistics_dict["fraction_of_very_high_a_sentences"] = protocol_statistics_dict[
                                                                            "num_of_very_high_a_sentences"] / \
                                                                        protocol_statistics_dict[
                                                                            "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_low_a_sentences"] = len(
            [value for value in vad_dict["a_values"] if value < 0.3])
        protocol_statistics_dict["fraction_of_low_a_sentences"] = protocol_statistics_dict["num_of_low_a_sentences"] / \
                                                                  protocol_statistics_dict[
                                                                      "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_very_low_a_sentences"] = len(
            [value for value in vad_dict["a_values"] if value < 0.1])
        protocol_statistics_dict["fraction_of_very_low_a_sentences"] = protocol_statistics_dict[
                                                                           "num_of_very_low_a_sentences"] / \
                                                                       protocol_statistics_dict[
                                                                           "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_high_d_sentences"] = len(
            [value for value in vad_dict["d_values"] if value > 0.7])
        protocol_statistics_dict["fraction_of_high_d_sentences"] = protocol_statistics_dict["num_of_high_d_sentences"] / \
                                                                   protocol_statistics_dict[
                                                                       "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_very_high_d_sentences"] = len(
            [value for value in vad_dict["d_values"] if value > 0.9])
        protocol_statistics_dict["fraction_of_very_high_d_sentences"] = protocol_statistics_dict[
                                                                            "num_of_very_high_d_sentences"] / \
                                                                        protocol_statistics_dict[
                                                                            "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_low_d_sentences"] = len(
            [value for value in vad_dict["d_values"] if value < 0.3])
        protocol_statistics_dict["fraction_of_low_d_sentences"] = protocol_statistics_dict["num_of_low_d_sentences"] / \
                                                                  protocol_statistics_dict[
                                                                      "num_of_sentences_in_protocol"]
        protocol_statistics_dict["num_of_very_low_d_sentences"] = len(
            [value for value in vad_dict["d_values"] if value < 0.1])
        protocol_statistics_dict["fraction_of_very_low_d_sentences"] = protocol_statistics_dict[
                                                                           "num_of_very_low_d_sentences"] / \
                                                                       protocol_statistics_dict[
                                                                           "num_of_sentences_in_protocol"]
        protocols_vad_statistics.append(protocol_statistics_dict)
    return protocols_vad_statistics


def temp_change_names_of_shards_to_fit_original(vad_shards_dir, name_file):
    shard_files = os.listdir(vad_shards_dir)
    with open(name_file) as f:
        texts = f.readlines()
    for line in texts:
        file_to_change = line.split()[1]
        new_name = line.split()[6]
        if new_name not in shard_files:
            old_file = os.path.join(vad_shards_dir, file_to_change)
            new_file = os.path.join(vad_shards_dir, new_name)
            os.rename(old_file, new_file)
        else:
            old_file = os.path.join(vad_shards_dir, file_to_change)
            new_file = os.path.join(vad_shards_dir, f'{new_name}_new')
            os.rename(old_file, new_file)
    new_shard_files = os.listdir(vad_shards_dir)
    for file in new_shard_files:
        if "new" in file:
            real_name = file.split("_new")[0]
            if real_name not in new_shard_files:
                old_file = os.path.join(vad_shards_dir, file)
                new_file = os.path.join(vad_shards_dir, real_name)
                os.rename(old_file, new_file)


def find_all_committee_session_sentences_in_shards(session_name_strings, input_dir_path, output_dir_path):
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_shard_size = 10000
    relevant_sentences = []
    shards = os.listdir(input_dir_path)

    output_shard_number = 0
    for shard in shards:
        if "shard" in shard:
            shard_num = int(shard.split("shard_")[1].split(".jsonl")[0])
        else:
            continue
        # if shard_num<39:
        #     continue
        print(f'processing {shard}')
        shard_path = os.path.join(input_dir_path, shard)
        shard_relevant_sentences = get_shard_session_sentences(shard_path, session_name_strings)
        if not shard_relevant_sentences:
            print(f'{shard} has no relevant sentences')
            continue

        relevant_sentences.extend(shard_relevant_sentences)
        while len(relevant_sentences) >= output_shard_size:
            shard_to_save = relevant_sentences[:output_shard_size]
            relevant_sentences = relevant_sentences[output_shard_size:]
            output_file_path = os.path.join(output_dir_path, f"relevant_sentences_shard_{output_shard_number}.jsonl")
            with open(output_file_path, 'w', encoding="utf-8") as f:
                for sentence in shard_to_save:
                    f.write(json.dumps(sentence, ensure_ascii=False) + '\n')
            output_shard_number += 1

    if relevant_sentences:
        output_file_path = os.path.join(output_dir_path, f"relevant_sentences_shard_{output_shard_number}.jsonl")
        with open(output_file_path, 'w', encoding="utf-8") as f:
            for sentence in relevant_sentences:
                f.write(json.dumps(sentence, ensure_ascii=False) + '\n')


def predict_vad_values_for_shard(sent_embeddings_model, shard_path, output_dir_path, shard_number):
    output_file_path = os.path.join(output_dir_path, f"vad_shard_{shard_number}.jsonl")
    if os.path.exists(output_file_path):
        print(f'output file already exists in dir {output_file_path}')
        return
    with open(shard_path, 'r', encoding="utf-8") as f:
        sentences_data = [json.loads(line) for line in f]

    sentences_texts = [sent["sentence_text"] for sent in sentences_data]
    sentences = process_input_texts_for_multi_model(sentences_texts)

    # Predict VAD values for all sentences in the shard
    v_predictions, a_predictions, d_predictions = predict_vad_values_on_sentences(sentences, sent_embeddings_model,
                                                                                  None)
    sentences_with_vad = add_vad_values_to_sentences_jsons(sentences_data, v_predictions, a_predictions, d_predictions)

    # Save the output shard
    save_output_shard(sentences_with_vad, output_dir_path, shard_number)

def check_which_shards_were_processed(relevant_sentences_shards_dir, vad_shards_dir, output_dir):
    processed_shards = []
    committee_sentences_shards = os.listdir(relevant_sentences_shards_dir)
    vad_shards = os.listdir(vad_shards_dir)
    for vad_shard in vad_shards:
        print(f'processing {vad_shard}')
        found_original_sentences_shard = False
        vad_shard_path = os.path.join(vad_shards_dir, vad_shard)
        vad_shard_num = int(vad_shard.split("shard_")[1].split(".jsonl")[0])
        vad_shard_df = pd.read_json(vad_shard_path, lines=True, encoding="utf-8")
        sentences_ids = list(vad_shard_df["_id"])
        for sent_shard in committee_sentences_shards:
            if sent_shard in processed_shards:
                continue
            sent_shard_path = os.path.join(relevant_sentences_shards_dir, sent_shard)
            sent_shard_df = pd.read_json(sent_shard_path, lines=True, encoding="utf-8")
            sent_shard_num = int(sent_shard.split("shard_")[1].split(".jsonl")[0])
            sent_shard_sentences_ids = list(sent_shard_df["_id"])
            if sent_shard_sentences_ids == sentences_ids:
                processed_shards.append(sent_shard)
                found_original_sentences_shard = True
                if vad_shard_num != sent_shard_num:
                    new_vad_shard_name = f'vad_shard_{sent_shard_num}.jsonl'
                    print(f'file {vad_shard} should be changed to {new_vad_shard_name}')

                break
        if not found_original_sentences_shard:
            print(f'{vad_shard} did not find original')
    non_processed_shards = [shard for shard in committee_sentences_shards if shard not in processed_shards]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for shard in non_processed_shards:
        original_shard_path = os.path.join(relevant_sentences_shards_dir, shard)
        output_shard_path = os.path.join(output_dir, shard)
        shutil.copy(original_shard_path, output_shard_path)

def predict_vad_values_of_session_sentences(input_dir_path, output_dir_path):
    print(f'started predict_vad_values_of_session_sentences', flush=True)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    try:
        shards = os.listdir(input_dir_path)
        if OUR_MULTI_PRETRAINED_MODEL:
            fine_tuned_model_reassembled = load_our_fine_tuned_multi_model()
            sent_embeddings_model = fine_tuned_model_reassembled
            tokenizer = None
        else:
            sent_embeddings_model = SentenceTransformer(SENT_TRANSFORMER_TYPE)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for shard in shards:
                shard_path = os.path.join(input_dir_path, shard)
                shard_num = int(shard.split("shard_")[1].split(".jsonl")[0])
                output_shard_number = shard_num
                futures.append(
                    executor.submit(predict_vad_values_for_shard, sent_embeddings_model, shard_path, output_dir_path, output_shard_number)
                )


            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing shard: {e}")

    except Exception as e:
        print(f"Error in processing: {e}")

    print("Process finished")

def print_highest_and_lowest_vad_sentences(vad_shards_dir, vad_column, filter_short_sentences=False, filter_non_hebrew = False):
    if vad_column == "v":
        vad_column = "valence"
    elif vad_column == "a":
        vad_column = "arousal"
    elif vad_column == "d":
        vad_column = "dominance"
    else:
        print(f'wrong vad_column: {vad_column}')
        return
    ten_highest_score_sentences = []
    ten_lowest_score_sentences = []
    for shard in os.listdir(vad_shards_dir):
        shard_path = os.path.join(vad_shards_dir, shard)
        with open(shard_path, encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            sent = json.loads(line)
            sent_text = sent["sentence_text"]
            if filter_short_sentences and len(sent_text.split())<2:
                continue
            if filter_non_hebrew and not is_hebrew(sent_text, print_non_hebrew=False):
                continue
            sent_score = sent["vad_values"][vad_column]
            if len(ten_highest_score_sentences) < 10:
                ten_highest_score_sentences.append({"sentence_text": sent["sentence_text"], "score": sent_score, "protocol_name":sent["protocol_name"], "speaker_name":sent["speaker_name"]})
                ten_highest_score_sentences = sorted(ten_highest_score_sentences, key=itemgetter('score'), reverse=True)
            else:
                lowest_high_sent = ten_highest_score_sentences[-1]
                if sent_score > lowest_high_sent["score"]:
                    ten_highest_score_sentences.pop(-1)
                    ten_highest_score_sentences.append({"sentence_text":sent["sentence_text"], "score":sent_score, "protocol_name":sent["protocol_name"], "speaker_name":sent["speaker_name"]})
                    ten_highest_score_sentences = sorted(ten_highest_score_sentences, key=itemgetter('score'), reverse=True)
            if len(ten_lowest_score_sentences) < 20:
                ten_lowest_score_sentences.append({"sentence_text": sent["sentence_text"], "score": sent_score, "protocol_name":sent["protocol_name"], "speaker_name":sent["speaker_name"]})
                ten_lowest_score_sentences = sorted(ten_lowest_score_sentences, key=itemgetter('score'))
            else:
                highest_low_sent = ten_lowest_score_sentences[-1]
                if sent_score < highest_low_sent["score"]:
                    ten_lowest_score_sentences.pop(-1)
                    ten_lowest_score_sentences.append({"sentence_text": sent["sentence_text"], "score": sent_score, "protocol_name":sent["protocol_name"], "speaker_name":sent["speaker_name"]})
                    ten_lowest_score_sentences = sorted(ten_lowest_score_sentences, key=itemgetter('score'))
    print(f'{vad_column} - ten highest scores sentences are:')
    for sent in ten_highest_score_sentences:
        print(f'{sent["sentence_text"]}, score: {round(sent["score"], 4)}')
    print(f'{vad_column} - twenty lowest scores sentences are:')
    for sent in ten_lowest_score_sentences:
        print(f'{sent["sentence_text"]}, score: {round(sent["score"], 4)}')

def calculate_protocols_male_female_numbers_dict(protocols_male_female_values):
    protocols_male_female_numbers_list = []
    for protocol, male_female_dict in protocols_male_female_values.items():
        protocol_male_female_numbers_dict = {}
        protocol_male_female_numbers_dict["protocol_name"] = protocol
        protocol_male_female_numbers_dict["num_male_sentences"] = male_female_dict["num_male_sentences"]
        protocol_male_female_numbers_dict["num_female_sentences"] = male_female_dict["num_female_sentences"]
        protocol_male_female_numbers_dict["num_male_members"] = len(male_female_dict["male_members"])
        protocol_male_female_numbers_dict["num_female_members"] = len(male_female_dict["female_members"])
        protocols_male_female_numbers_list.append(protocol_male_female_numbers_dict)
    return protocols_male_female_numbers_list
def calculate_protocols_coalition_opposition_numbers_dict(protocols_coalition_opposition_values):
    protocols_coalition_opposition_numbers_list = []
    for protocol, coalition_opposition_dict in protocols_coalition_opposition_values.items():
        protocol_coalition_opposition_numbers_dict = {}
        protocol_coalition_opposition_numbers_dict["protocol_name"] = protocol
        protocol_coalition_opposition_numbers_dict["num_coalition_sentences"] = coalition_opposition_dict["num_coalition_sentences"]
        protocol_coalition_opposition_numbers_dict["num_opposition_sentences"] = coalition_opposition_dict["num_opposition_sentences"]
        protocol_coalition_opposition_numbers_dict["num_coalition_members"] = len(coalition_opposition_dict["coalition_members"])
        protocol_coalition_opposition_numbers_dict["num_opposition_members"] = len(coalition_opposition_dict["opposition_members"])
        protocols_coalition_opposition_numbers_list.append(protocol_coalition_opposition_numbers_dict)
    return protocols_coalition_opposition_numbers_list
def count_num_of_coalition_opposition_speakers(relevant_sentences_dir_path, output_path):
    shards = os.listdir(relevant_sentences_dir_path)
    protocols_coalition_opposition_values = {}
    for shard in shards:
        shard_path = os.path.join(relevant_sentences_dir_path, shard)
        with open(shard_path, encoding="utf-8") as shard_file:
            for line in shard_file:
                sent_entity = json.loads(line)
                speaker_id = sent_entity["speaker_id"]
                if "member_of_coalition_or_opposition" in sent_entity:
                    member_of_coalition_or_opposition = sent_entity["member_of_coalition_or_opposition"]
                else:
                    member_of_coalition_or_opposition = None
                protocol_name = sent_entity["protocol_name"]
                protocol_values_dict = protocols_coalition_opposition_values.get(protocol_name, None)
                if not protocol_values_dict:
                    num_of_coalition_sents = 0
                    num_of_opposition_sents = 0
                    coalition_members = set()
                    opposition_members = set()
                    protocols_coalition_opposition_values[protocol_name] = {"num_coalition_sentences": num_of_coalition_sents, "num_opposition_sentences": num_of_opposition_sents, "coalition_members": coalition_members, "opposition_members": opposition_members}
                    protocol_values_dict = protocols_coalition_opposition_values[protocol_name]
                if member_of_coalition_or_opposition == "coalition":
                    protocol_values_dict["num_coalition_sentences"] += 1
                    protocol_values_dict["coalition_members"].add(speaker_id)
                elif member_of_coalition_or_opposition == "opposition":
                    protocol_values_dict["num_opposition_sentences"] += 1
                    protocol_values_dict["opposition_members"].add(speaker_id)
                else:
                    continue
    protocols_coalition_opposition_values_numbers = calculate_protocols_coalition_opposition_numbers_dict(protocols_coalition_opposition_values)
    df = pd.DataFrame.from_records(protocols_coalition_opposition_values_numbers)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    df.to_csv(output_path, index=False)

def count_num_of_male_female_speakers(relevant_sentences_dir_path, output_path):
    shards = os.listdir(relevant_sentences_dir_path)
    protocols_male_female_values = {}
    for shard in shards:
        shard_path = os.path.join(relevant_sentences_dir_path, shard)
        with open(shard_path, encoding="utf-8") as shard_file:
            for line in shard_file:
                sent_entity = json.loads(line)
                speaker_id = sent_entity["speaker_id"]
                if "speaker_gender" in sent_entity:
                    speaker_gender = sent_entity["speaker_gender"]
                else:
                    speaker_gender = None
                protocol_name = sent_entity["protocol_name"]
                protocol_values_dict = protocols_male_female_values.get(protocol_name, None)
                if not protocol_values_dict:
                    num_of_male_sents = 0
                    num_of_female_sents = 0
                    male_members = set()
                    female_members = set()
                    protocols_male_female_values[protocol_name] = {"num_male_sentences": num_of_male_sents, "num_female_sentences": num_of_female_sents, "male_members": male_members, "female_members": female_members}
                    protocol_values_dict = protocols_male_female_values[protocol_name]
                if speaker_gender == "male":
                    protocol_values_dict["num_male_sentences"] += 1
                    protocol_values_dict["male_members"].add(speaker_id)
                elif speaker_gender == "female":
                    protocol_values_dict["num_female_sentences"] += 1
                    protocol_values_dict["female_members"].add(speaker_id)
                else:
                    continue
    protocols_male_female_values_numbers = calculate_protocols_male_female_numbers_dict(protocols_male_female_values)
    df = pd.DataFrame.from_records(protocols_male_female_values_numbers)
    df['protocol_name'] = df['protocol_name'].astype(str)
    df = df.sort_values(by="protocol_name")
    df.to_csv(output_path, index=False)


if __name__ == '__main__':

    # session_name_strings = ["הוועדה לקידום מעמד האישה ולשוויון מגדרי", "הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה", " הוועדה לקידום מעמד האישה ולשוויון מגדרי" ]
    # session_name_strings = ["ועדת החוץ והביטחון", " ועדת החוץ והביטחון"]
    # session_name_strings = ["ועדת הכספים", " ועדת הכספים", " ועדת הכספים "]
    # session_name_strings = ["ועדת הכלכלה", " ועדת הכלכלה"]
    # session_name_strings = ["ועדת החוקה, חוק ומשפט", " ועדת החוקה חוק ומשפט"]
    # session_name_strings = ["ועדת החינוך, התרבות והספורט", "ועדת החינוך והתרבות", " ועדת החינוך, התרבות והספורט"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות", "ועדת הבריאות","ועדת העבודה והרווחה"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות","ועדת הבריאות"]
    # session_name_strings = ["ועדת העבודה, הרווחה והבריאות","ועדת העבודה והרווחה"]
    # session_name_strings = ["ועדת הפנים והגנת הסביבה","ועדת הפנים ואיכות הסביבה", " ועדת הפנים והגנת הסביבה"]
    # session_name_strings = ["הוועדה לענייני ביקורת המדינה"," הוועדה לענייני ביקורת המדינה"]
    # session_name_strings = ["ועדת העלייה, הקליטה והתפוצות"]
    # session_name_strings = ["ועדת הכנסת", " ועדת הכנסת"]
    # session_name_strings = ["הוועדה המיוחדת לזכויות הילד", "הוועדה המיוחדת לקידום מעמד הילד"]
    # session_name_strings = ["הוועדה המיוחדת לפניות הציבור"]
    session_name_strings = ["ועדת המדע והטכנולוגיה", "הוועדה המיוחדת לענייני מחקר ופיתוח מדעי וטכנולוגי"]
    # session_name_strings = ["הוועדה המיוחדת לבחינת בעיית העובדים הזרים", "הוועדה המיוחדת לעובדים זרים"]
    # session_name_strings = ["הוועדה המיוחדת ליישום הנגשת המידע הממשלתי ועקרונות שקיפותו לציבור"]
    # session_name_strings = ["הוועדה המיוחדת למאבק בנגעי הסמים והאלכוהול", "הוועדה המיוחדת למאבק בנגע הסמים", "הוועדה המיוחדת לענייני התמכרויות, סמים ואתגרי הצעירים בישראל", "הוועדה המיוחדת להתמודדות עם סמים ואלכוהול"]
    # session_name_strings = ["ועדת המשנה לתקנות שוויון זכויות לאנשים עם מוגבלות", "ועדת המשנה ליישום חוק שוויון זכויות לאנשים עם מוגבלות"]
    # session_name_strings = ["ועדת ביטחון הפנים"]
    # session_name_strings = ["הוועדה המיוחדת לצדק חלוקתי ולשוויון חברתי"]
    # session_name_strings = ["הוועדה המסדרת"]
    # session_name_strings = ["ישיבת מליאה"]

    input_dir_path = os.path.join(processed_knesset_data_path,"sentences_jsonl_files\\committee_full_sentences_shards")
    plenary_input_dir_path = os.path.join(processed_knesset_data_path,"sentences_jsonl_files\\plenary_full_sentences_shards")
    vad_values_committees_dir = os.path.join(processed_knesset_data_path,"sentences_jsonl_files\\sentences_with_vad_values\\committees")
    print(f'working on {session_name_strings[0]}')

    # session_english_name = "women_status"
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
    session_english_name = "technology"
    # session_english_name = "foreign_workers"
    # session_english_name = "access_to_information"
    # session_english_name = "drugs"
    # session_english_name = "people_with_disabilities"
    # session_english_name = "inner_securitiy"
    # session_english_name = "justice_social_equality"
    # session_english_name = "organizing"
    # session_english_name = "plenary"



    this_session_dir = os.path.join(vad_values_committees_dir, session_english_name)
    vad_output_dir_path = os.path.join(this_session_dir, "vad_shards")
    # temp_change_names_of_shards_to_fit_original(vad_output_dir_path, "tmp_name_changes_funds.txt")
    # print_highest_and_lowest_vad_sentences(vad_output_dir_path, "v", filter_short_sentences=True, filter_non_hebrew = True)
    # print_highest_and_lowest_vad_sentences(vad_output_dir_path, "a", filter_short_sentences=True, filter_non_hebrew = True)
    # print_highest_and_lowest_vad_sentences(vad_output_dir_path, "d", filter_short_sentences=True, filter_non_hebrew = True)


    relevant_sentences_dir_path = os.path.join(this_session_dir,"sentences_shards")

    # find_all_committee_session_sentences_in_shards(session_name_strings, input_dir_path, output_dir_path=relevant_sentences_dir_path)
    # predict_vad_values_of_session_sentences(relevant_sentences_dir_path, vad_output_dir_path)
    print(f'finished predicting vad')
    coalition_opposition_number_path = os.path.join(this_session_dir, f"{session_english_name}_coalition_opposition_number.csv")
    count_num_of_coalition_opposition_speakers(relevant_sentences_dir_path, coalition_opposition_number_path)
    male_female_number_path = os.path.join(this_session_dir, f"{session_english_name}_male_female_number.csv")
    count_num_of_male_female_speakers(relevant_sentences_dir_path, male_female_number_path)
    session_stats_path = os.path.join(this_session_dir, f"{session_english_name}_vad_stats.csv")
    create_vad_avg_and_median_per_protocol_csv(vad_output_dir_path, session_stats_path,filter_short_sentences=True, filter_non_hebrew=True)
    print(f'vad statistics of {session_english_name} all protocols')
    df = pd.read_csv(session_stats_path)
    for col_name, column_values in df.items():
        if col_name == "protocol_name" or "num" in col_name:
            continue
        apply_Mann_Kendall_Test(list(column_values), significance_level=0.05, data_name=col_name)



    coalition_session_stats_path = os.path.join(this_session_dir, f"{session_english_name}_coalition_vad_stats.csv")
    opposition_session_stats_path = os.path.join(this_session_dir, f"{session_english_name}_opposition_vad_stats.csv")
    create_vad_avg_and_median_per_coalition_opposition_csvs(vad_output_dir_path, coalition_session_stats_path, opposition_session_stats_path,filter_short_sentences=True, filter_non_hebrew=True)
    print(f'vad statistics of {session_english_name} coalition')
    coalition_df = pd.read_csv(coalition_session_stats_path)
    for col_name, column_values in coalition_df.items():
        if col_name == "protocol_name" or "num" in col_name:
            continue
        apply_Mann_Kendall_Test(list(column_values), significance_level=0.05, data_name=col_name)

    print(f'vad statistics of {session_english_name} opposition')
    opposition_df = pd.read_csv(opposition_session_stats_path)
    for col_name, column_values in opposition_df.items():
        if col_name == "protocol_name" or "num" in col_name:
            continue
        apply_Mann_Kendall_Test(list(column_values), significance_level=0.05, data_name=col_name)
    common_protocols = pd.merge(coalition_df[['protocol_name']], opposition_df[['protocol_name']], on='protocol_name')
    print(f'performing t-test between coalition and opposition')
    # Filter original dataframes based on common protocol names
    df1_filtered = coalition_df[coalition_df['protocol_name'].isin(common_protocols['protocol_name'])]
    df2_filtered = opposition_df[opposition_df['protocol_name'].isin(common_protocols['protocol_name'])]
    coalition_df = df1_filtered
    opposition_df = df2_filtered
    for (coalition_col_name, coalition_col_values), (opposition_col_name, opposition_col_values) in zip(coalition_df.items(), opposition_df.items()):
        if ("_d_" in coalition_col_name or "_v_" in coalition_col_name or "_a_" in coalition_col_name) and "num" not in coalition_col_name:
            perform_t_test(coalition_col_values, opposition_col_values, f'coalition_{coalition_col_name}', f'opposition_{opposition_col_name}')
