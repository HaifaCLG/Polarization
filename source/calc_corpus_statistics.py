import json
import os

committees_path = os.path.join(processed_knesset_data_path, "sentences_jsonl_files", "sentences_with_vad_values", "committees")

total_num_of_sentences = 0
total_num_of_tokens = 0

committee_dirs = os.listdir(committees_path)
for committee_dir in committee_dirs:
    committee_dir_path = os.path.join(committees_path, committee_dir)
    if not os.path.isdir(committee_dir_path):
        continue
    sentences_shards_dir = os.path.join(committee_dir_path, "sentences_shards")
    num_of_sentences_in_committee = 0
    num_of_tokens_in_committee= 0
    for shard in os.listdir(sentences_shards_dir):
        shard_path = os.path.join(sentences_shards_dir, shard)
        try:
            with open(shard_path, encoding="utf-8") as file:
                for line in file:
                    num_of_sentences_in_committee += 1
                    try:
                        sent_entity = json.loads(line)
                    except Exception as e:
                        print(f'couldnt load json: {line}. error was: {e}')
                        continue
                    sent_text = sent_entity["sentence_text"]
                    tokens = sent_text.split()
                    num_of_tokens_in_sent = len(tokens)
                    num_of_tokens_in_committee += num_of_tokens_in_sent
        except Exception as e:
            print(f'couldnt open shard: {shard}. exception was: {e}')
            continue
    print(f'in committee: {committee_dir}')
    print(f'num of sentences is: {num_of_sentences_in_committee}')
    print(f'num of tokens is: {num_of_tokens_in_committee}')
    total_num_of_tokens += num_of_tokens_in_committee
    total_num_of_sentences += num_of_sentences_in_committee


print(f'total number of sentences: {total_num_of_sentences}')
print(f'total number of tokens: {total_num_of_tokens}')




