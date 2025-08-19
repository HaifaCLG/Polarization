import json
import os

import pandas as pd

from statistic_functions import apply_Mann_Kendall_Test


from transformers import pipeline
from HeBERT.src.HebEMO import *


def predict_and_save_sentiment_and_score_shards(committee_path):
    oracle = pipeline('sentiment-analysis', model='dicta-il/dictabert-sentiment')
    committee_sents_path = os.path.join(committee_path, "sentences_shards")
    shards_names = os.listdir(committee_sents_path)
    for shard in shards_names:
        shard_path = os.path.join(committee_sents_path, shard)
        shard_output_path = os.path.join(committee_path, "sentiment_and_emotions_shards", shard)
        if os.path.exists(shard_output_path):
            print(f'{shard} already exits. moving on')
            continue
        predict_and_save_sentiment_and_emotion_scores_for_full_shard(shard_path, shard_output_path, oracle)
def predict_and_save_sentiment_and_emotion_scores_for_full_shard(shard_path, shard_output_path, oracle):

    try:
        with open(shard_path, encoding="utf-8") as shard_file:
            sentences = [json.loads(line) for line in shard_file]
    except Exception as e:
        print(f'couldnt open shard: {shard_path}. Error: {e}')
    sentences_texts = [sent["sentence_text"] for sent in sentences]
    try:
        sentiment_predictions = oracle(sentences_texts)
    except Exception as e:
        print(f'error in dicta sentiment prediction: {e}')
        return
    for sent, prediction in zip(sentences, sentiment_predictions):
        label = prediction['label']
        sent["dicta-sentiment"] = label
    FULL_EMOTIONS = False
    if FULL_EMOTIONS:
        HebEMO_model = HebEMO(device=0)
        hebEMO_df = HebEMO_model.hebemo(sentences_texts, save_results=True)
        for sent, (idx, emotions_row) in zip(sentences, hebEMO_df.iterrows()):
            sent["hebEMO-anticipation"] = emotions_row["anticipation"]
            sent["hebEMO-joy"] = emotions_row["joy"]
            sent["hebEMO-trust"] = emotions_row["trust"]
            sent["hebEMO-fear"] = emotions_row["fear"]
            sent["hebEMO-surprise"] = emotions_row["surprise"]
            sent["hebEMO-anger"] = emotions_row["anger"]
            sent["hebEMO-sadness"] = emotions_row["sadness"]
            sent["hebEMO-disgust"] = emotions_row["disgust"]

    os.makedirs(os.path.dirname(shard_output_path), exist_ok=True)
    try:
        with open(shard_output_path, "w", encoding="utf-8") as output_file:
            for sent in sentences:
                output_file.write(json.dumps(sent, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f'Couldn’t write to output shard: {shard_output_path}. Error: {e}')

def count_percantage_of_sentiment_sentences_per_protocol(committee_sents_path, output_path):
    protocols_dict = {}#{protocol_name:{num_of_positive:X,num_of_negative:X, num_of_neutral:X, total:X}}
    shards_names = os.listdir(committee_sents_path)
    # oracle = pipeline('sentiment-analysis', model='dicta-il/dictabert-sentiment')

    for shard in shards_names:
        try:
            with open(os.path.join(committee_sents_path, shard), encoding="utf-8") as shard_file:
                sentences = shard_file.readlines()
        except Exception as e:
            print(f'couldnt open shard: {shard}. Error: {e}')
            continue
        for sent in sentences:
            try:
                sent_entity = json.loads(sent)
            except Exception as e:
                print(f'couldnt load sentence in {shard}. Error:{e}')
                continue
            sent_text = sent_entity["sentence_text"]
            sentiment_label = sent_entity["dicta-sentiment"]
            anger_label = sent_entity["hebEMO-anger"]
            sent_protocol = sent_entity["protocol_name"]
            protocol_dict = protocols_dict.get(sent_protocol,None)
            if not protocol_dict:
                protocols_dict[sent_protocol] = {}
                protocol_dict = protocols_dict[sent_protocol]
                protocol_dict["num_positive"] = 0
                protocol_dict["num_negative"] = 0
                protocol_dict["num_neutral"] = 0
                protocol_dict["total"] = 0
                protocol_dict["num_anger"] = 0

            if sentiment_label == "Positive":
                protocol_dict["num_positive"] += 1
            elif sentiment_label == "Negative":
                protocol_dict["num_negative"] += 1
            elif sentiment_label == "Neutral":
                protocol_dict["num_neutral"] += 1
            else:
                print(f'wrong label: {sentiment_label}')
            protocol_dict["total"] += 1
            if anger_label == 1:
                protocol_dict["num_anger"] += 1

            assert protocol_dict["num_positive"] + protocol_dict["num_negative"] +protocol_dict["num_neutral"] == protocol_dict["total"]

    for protocol_name, protocol_values in protocols_dict.items():
        if protocol_values["total"] != 0:
            perc_positive = protocol_values["num_positive"]/protocol_values["total"]
            perc_negative = protocol_values["num_negative"]/protocol_values["total"]
            protocol_values["perc_positive"] = perc_positive
            protocol_values["perc_negative"] = perc_negative
            perc_anger = protocol_values["num_anger"]/protocol_values["total"]
            protocol_values["perc_anger"] = perc_anger

    df = pd.DataFrame.from_dict(protocols_dict, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'protocol_name'}, inplace=True)
    df.sort_values(by='protocol_name', inplace=True)
    df.to_csv(output_path, index=False)
    print(f'saved to csv ')



if __name__ == '__main__':
    committee_name = "economy"
    committee_path = os.path.join(processed_knesset_data_path, "sentences_jsonl_files", "sentences_with_vad_values",
                                   "committees", committee_name)
    predict_and_save_sentiment_and_score_shards(committee_path)
    output_path = os.path.join(committee_path,f"{committee_name}_sentiment_numbers.csv")
    chosen_committee_path = os.path.join(committee_path,"sentiment_and_emotions_shards")

    count_percantage_of_sentiment_sentences_per_protocol(chosen_committee_path, output_path)

    df = pd.read_csv(output_path)
    df = df.sort_values(by="protocol_name")
    for col_name, column_values in df.items():
        if col_name == "protocol_name" or "num" in col_name or "total" in col_name:
            continue
        apply_Mann_Kendall_Test(list(column_values), significance_level=0.05, data_name=col_name)
