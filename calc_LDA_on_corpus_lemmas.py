import json
import os
import random
from datetime import datetime
from pprint import pprint

import gensim as gensim
import pandas as pd
import pyLDAvis as pyLDAvis
from gensim import corpora
from gensim.models import CoherenceModel, LdaMulticore
from matplotlib import pyplot as plt

from aux_functions import save_object, load_object
os.environ['MALLET_HOME'] = 'C:\\Users\\gilis\\PycharmProjects\\extremism_on_knesset_corpus\\mallet\\mallet-2.0.8\\mallet-2.0.8'
RANDOM_SEED = 4
def get_hebrew_stop_words():
    with open("heb_stopwords_revised.txt", encoding="utf-8") as file:
        raw_stop_words =  file.readlines()
        stop_words = [word.strip() for word in raw_stop_words]
    return stop_words

import os
import json
import random

def get_turns_lemmas(paths, percent=1):
    num_of_none_morphological_fields_counter = 0
    turns_counter = 0
    all_turns_lemmas = []
    all_turns_ids = []
    stop_words = get_hebrew_stop_words()  # Assuming this function is defined elsewhere

    for path in paths:
        if os.path.isdir(path):
            protocol_files = os.listdir(path)
        elif os.path.isfile(path):
            protocol_file = os.path.basename(path)
            protocol_files = [protocol_file]
            path = os.path.dirname(path)
        for protocol_file_name in protocol_files:
            current_turn_num = 0
            turn_lemmas_list = []

            protocol_file_path = os.path.join(path, protocol_file_name)
            try:
                with open(protocol_file_path, encoding="utf-8") as f:
                    protocol = json.load(f)
                    protocol_sentences = protocol["protocol_sentences"]
            except Exception as e:
                print(f"Couldn't read file {protocol_file_path}. Error: {e}")
                continue

            for sent in protocol_sentences:
                sent_id = sent["sentence_id"]
                protocol_name = sent["protocol_name"]
                turn_num = sent["turn_num_in_protocol"]

                if turn_num != current_turn_num:
                    if turn_lemmas_list:  # If there's data to save from the previous turn
                        all_turns_lemmas.append(turn_lemmas_list)
                        all_turns_ids.append(f"{protocol_name}#{current_turn_num}")
                        turns_counter += 1
                    current_turn_num = turn_num
                    turn_lemmas_list = []

                sent_lemma_list = create_list_of_lemmas_from_sentence(sent, num_of_none_morphological_fields_counter, stop_words_list=stop_words)
                turn_lemmas_list.extend(sent_lemma_list)

            if turn_lemmas_list:  # Ensure the last turn in the file is also processed
                all_turns_lemmas.append(turn_lemmas_list)
                all_turns_ids.append(f"{protocol_name}#{current_turn_num}")
                turns_counter += 1

    # Report counts before sampling
    print(f'num_of_none_morphological_fields_counter is {num_of_none_morphological_fields_counter}')
    print(f'total number of turns is {turns_counter}')
    print(f'num of used turns before sampling {len(all_turns_lemmas)}')

    # Sampling
    sample_size = max(1, round(float(percent) * len(all_turns_lemmas)))  # Ensure at least one sample
    random_indices = random.sample(range(len(all_turns_lemmas)), sample_size)
    sampled_turns_lemmas = [all_turns_lemmas[i] for i in random_indices]
    sampled_turns_ids = [all_turns_ids[i] for i in random_indices]

    print(f'final num of used sentences {len(sampled_turns_lemmas)}')
    return sampled_turns_lemmas, sampled_turns_ids

def get_sentences_lemmas(paths, percent = 1):
    num_of_none_morpholgical_fields_counter = 0
    sentences_counter = 0
    all_sents_lemmas = []
    all_sents_ids = []
    stop_words = get_hebrew_stop_words()
    for path in paths:
        shard_files = os.listdir(path)
        for shard_file_name in shard_files:
            shard_file_path = os.path.join(path, shard_file_name)
            try:
                with open(shard_file_path, encoding="utf-8") as f:
                    if not f:
                        print("hi")
                    for line in f:
                        try:
                            sent = json.loads(line)
                            sent_id = sent["sentence_id"]
                            protocol_name = sent["protocol_name"]
                        except Exception as e:
                            print(f'couldnt load json in file {shard_file_path}. line was: {line}')
                            continue
                        sentences_counter += 1
                        sent_lemma_list = create_list_of_lemmas_from_sentence(sent, num_of_none_morpholgical_fields_counter, stop_words_list=stop_words)
                        if sent_lemma_list:
                            all_sents_lemmas.append(sent_lemma_list)
                            all_sents_ids.append(f"{protocol_name}#{sent_id}")
                            # print(f"{protocol_name}#{sent_id} : {sent['sentence_text']}")
            except Exception as e:
                print(f"couldnt read file {shard_file_path}. error was: {e}")
                continue
    print(f'num_of_none_morpholgical_fields_counter is {num_of_none_morpholgical_fields_counter}')
    print(f'total number of sentences is {sentences_counter}')
    print(f' num of used sentences before sampling {len(all_sents_lemmas)}')
    # Zip the lemmas and ids together to maintain their pairing
    zipped_lemmas_ids = list(zip(all_sents_lemmas, all_sents_ids))

    # Calculate sample size
    sample_size = round(float(percent) * len(zipped_lemmas_ids))

    # Sample from the zipped list
    sampled_zipped_lemmas_ids = random.sample(zipped_lemmas_ids, sample_size)

    # Unzip the sampled pairs back into two lists
    sampled_sents_lemmas, sampled_sents_ids = zip(*sampled_zipped_lemmas_ids)

    # Convert zipped objects back into lists if you need to use them as lists
    sampled_sents_lemmas = list(sampled_sents_lemmas)
    sampled_sents_ids = list(sampled_sents_ids)

    print(f'final num of used sentences {len(sampled_sents_lemmas)}')

    return sampled_sents_lemmas, sampled_sents_ids
def create_list_of_lemmas_from_sentence(sent,num_of_none_morpholgical_fields_counter,  allowed_postags= ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'], stop_words_list=None):#
    sent_lemmas = []
    sent_morph_fields = sent["morphological_fields"]
    if sent_morph_fields:
        for item in sent_morph_fields:
            if item["upos"] in allowed_postags:
                if item["lemma"]:
                    if item["lemma"] in stop_words_list:
                        continue
                    sent_lemmas.append(item["lemma"])
    else:
        num_of_none_morpholgical_fields_counter +=1
    return sent_lemmas


def make_bigrams(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words]

def make_trigrams(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in data_words]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, random_seed=RANDOM_SEED, prefix="mallet")
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def row_makeover_for_print(row):
    new_row = []
    for tuple in row[:5]:
        topic_num = tuple[0]
        if tuple[1]>=0.02:
            prob = f'{round(tuple[1]*100, 3)}%'
            new_tup = (topic_num, prob)
            new_row.append(new_tup)
        else:
            break
    return new_row


def get_original_text_from_text_id(text_id, text_lemmas):
    turns = False
    protocol_name = text_id.split("#")[0]
    num = text_id.split("#")[1]
    if len(num)>10:
        sent_id = num
    else:
        turn_num = int(num)
        turns = True
    if "ptm" in protocol_name:
        protocol_path = os.path.join("D:\\data\\gili\\processed_knesset\\protocols\\plenary_protocols\\plenary_protocols_jsons",f"{protocol_name}.jsonl")
    elif "ptv" in protocol_name:
        protocol_path = os.path.join(
            "D:\\data\\gili\\processed_knesset\\protocols\\committee_protocols\\committee_protocols_jsons",
            f"{protocol_name}.jsonl")
    else:
        print(f'wrong type of protocol: {protocol_name}')
    with open(protocol_path, encoding="utf-8") as protocol_file:
        protocol = json.load(protocol_file)
        sentences = protocol['protocol_sentences']
    if turns:
        text = ""
        found_turn = False
        for sent in sentences:
            if sent["turn_num_in_protocol"]== turn_num:
                found_turn = True
                text += f'{sent["sentence_text"]} '
            else:
                if found_turn:
                    break
        if not found_turn:
            print(f'turn number {turn_num} was not found in protocol {protocol_name} in id: {text_id}')
    else:

        for sent in sentences:
            if sent["sentence_id"] == sent_id:
                text = sent["sentence_text"]
    return text


def get_topics_for_documents(ldamodel, corpus, texts, text_ids,filename, chunksize=100):
    pretty_print = False
    progress_filename = f"{filename}_progress.txt"

    # Check for existing progress and set the starting index
    try:
        with open(progress_filename, 'r') as f:
            start_index = int(f.read().strip()) + 1
            print(f"Resuming from index {start_index}")
    except FileNotFoundError:
        start_index = 0
        # If starting from beginning, write headers to the file
        pd.DataFrame(columns=['text_id', 'Dominant_Topic', 'Perc_Contribution', 'top_5_topics']).to_csv(filename,
                                                                                                        index=False)

    for i in range(start_index, len(corpus), chunksize):
        chunk = corpus[i:i + chunksize]
        topic_values = []  # Reset topic values for each chunk

        for j, row_list in enumerate(ldamodel[chunk]):
            row = sorted(row_list, key=lambda x: (x[1]), reverse=True)
            if len(row) > 0:
                topic_num, prop_topic = row[0]
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                adjusted_index = i + j

                if adjusted_index < len(texts):
                    text_id = text_ids[adjusted_index]
                    text_lemmas = texts[adjusted_index]


                    pretty_row = row_makeover_for_print(row)
                    if pretty_print:
                        text = get_original_text_from_text_id(text_id, text_lemmas)
                        if len(text_lemmas) > 15:  # Example condition for pretty print
                            print(f'Text ID: {text_id}, Lemmas: {text_lemmas}', flush=True)
                            print(f'Original Text: {text}', flush=True)
                            print(f'Pretty Row: {pretty_row}', flush=True)

                    topic_values.append([text_id, int(topic_num), round(prop_topic, 4), pretty_row])

        # Append current chunk's DataFrame to CSV, without headers if resuming
        pd.DataFrame(topic_values, columns=['text_id', 'Dominant_Topic', 'Perc_Contribution', 'top_5_topics']).to_csv(
            filename, mode='a', header=False, index=False)

        # Update progress file with the last index of this chunk
        with open(progress_filename, 'w') as f:
            f.write(str(i + chunksize - 1))

        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Finished processing chunk up to index {i + chunksize - 1}. Current time: {current_time}", flush=True)

    print('Finished processing all documents. You may now resume from the last checkpoint if needed.', flush=True)


def format_topics_sentences(ldamodel, corpus, texts, chunksize=100):
    topic_values = []
    for i in range(0, len(corpus), chunksize):
        chunk = corpus[i:i + chunksize]
        for j, row_list in enumerate(ldamodel[chunk]):
            row = sorted(row_list, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution, and Keywords for each document
            if len(row) > 0:
                topic_num, prop_topic = row[0]
                wp = ldamodel.show_topic(topic_num)
                # topic_keywords = ", ".join([word for word, prop in wp])
                # Make sure to adjust the index correctly when accessing 'texts'
                adjusted_index = i + j
                if adjusted_index < len(texts):
                    text = texts[adjusted_index]
                    print(text, flush=True)
                    print(row, flush=True)
                    # topic_values.append([int(topic_num), round(prop_topic, 4), topic_keywords, text])
                    topic_values.append([int(topic_num), round(prop_topic, 4), text])


    # Create DataFrame
    df = pd.DataFrame(topic_values, columns=['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Text'])
    return df


def prepare_texts_for_lda(paths, prefix, id2word_prefix, percent, create_new_lemmas, create_new_lemma_bigrams, create_new_id_2_word, create_new_corpus, turns=False, text_ids=False):

    if create_new_lemma_bigrams:
        if create_new_lemmas:
            if turns:
                text_lemmas, text_ids = get_turns_lemmas(paths, percent)
            else:
                text_lemmas, text_ids = get_sentences_lemmas(paths, percent)
            save_object(text_lemmas, f"{prefix}knesset_corpus_lemmas")
            save_object(text_ids, f"{prefix}knesset_corpus_text_ids")
        else:
            text_lemmas = load_object(f"{prefix}knesset_corpus_lemmas")
            text_ids = load_object(f"{prefix}knesset_corpus_text_ids")
        lemmas_bigrams = make_bigrams(text_lemmas)
        save_object(lemmas_bigrams, f"{prefix}knesset_lemmas_bigrams")
    else:
        lemmas_bigrams = load_object(f"{prefix}knesset_lemmas_bigrams")
        if text_ids:
            text_ids = load_object(f"{prefix}knesset_corpus_text_ids")
        else:
            text_ids = []
    print(f'finished lemmas bigrams', flush=True)

    if create_new_id_2_word:
        id2word = corpora.Dictionary(lemmas_bigrams)
        save_object(id2word, f"{id2word_prefix}knesset_lemmas_bigrams_id2word")
    else:
        id2word = load_object(f"{id2word_prefix}knesset_lemmas_bigrams_id2word")
    print(f'finished id2word' , flush=True)
    texts = lemmas_bigrams
    if create_new_corpus:
        corpus = [id2word.doc2bow(text) for text in texts]
        # Check for empty documents
        empty_docs = [idx for idx, doc in enumerate(corpus) if not doc]
        print("Empty documents indices:", empty_docs)

        save_object(corpus, f"{prefix}knesset_corpus_for_lda")
    else:
        corpus = load_object(f"{prefix}knesset_corpus_for_lda")
    print(f'finished corpus')
    return lemmas_bigrams, id2word, corpus, text_ids

def create_and_train_lda_model(mallet_path, corpus, id2word, number_of_topics, prefix, seed=0):
    lda_mallet_path = os.path.join("C:\\Users\\gilis\\PycharmProjects\\extremism_on_knesset_corpus\\mallet_lda_model", f'{prefix}_knesset_mallet_lda')
    if CREATE_NEW_LDA_MODEL:
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=number_of_topics,
                                                     id2word=id2word, random_seed=RANDOM_SEED)
        ldamallet.save(lda_mallet_path)
    else:
        ldamallet = gensim.models.wrappers.LdaMallet.load(lda_mallet_path)
        # Show Topics
    with open("lda_results.txt", "a", encoding="utf-8") as file:
        pprint(ldamallet.show_topics(formatted=False, num_topics=number_of_topics, num_words=number_of_words), file)
    pprint(ldamallet.show_topics(formatted=False, num_topics=number_of_topics, num_words=number_of_words))
    current_lda_model = ldamallet

    # Compute Coherence Score
    coherence_current_lda_model = CoherenceModel(model=current_lda_model, texts=texts, dictionary=id2word,
                                                 coherence='c_v')
    coherence_lda = coherence_current_lda_model.get_coherence()
    print(f"\n{prefix}Coherence Score: {coherence_lda}")
    with open("lda_results.txt", "a") as file:
        pprint(f"\n{prefix}Coherence Score: {coherence_lda}", file)
    return current_lda_model

def greedy_num_of_topics_search(id2word, texts, corpus):
    limit = 80
    start = 5
    step = 5
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus,
                                                            texts=texts, start=start, limit=limit, step=step)
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


if __name__ == '__main__':
    with open("lda_results.txt", "a") as file:
        pprint("results:", file)
    CREATE_NEW_LEMMAS = False
    CREATE_NEW_LEMMA_BIGRAMS = False
    CREATE_NEW_ID_2_WORD = False
    CREATE_NEW_CORPUS = False
    CREATE_NEW_LDA_MODEL = True
    MULTI_CORE_LDA = False
    Mallet = True
    number_of_topics = 50
    number_of_words = 25
    plenary_paths = "D:\\data\\gili\\processed_knesset\\sentences_jsonl_files\\plenary_full_sentences_shards"
    committee_paths ="D:\\data\\gili\\processed_knesset\\sentences_jsonl_files\\committee_full_sentences_shards"
    # all_paths = [committee_paths, plenary_paths]
    all_paths = [plenary_paths]
    greedy = False

    percent = 0.5

    prefix = f"plenary_{str(percent)}_{number_of_topics}_topics_"

    lemmas_bigrams, id2word, corpus, text_ids = prepare_texts_for_lda(all_paths, prefix,prefix,percent, CREATE_NEW_LEMMAS, CREATE_NEW_LEMMA_BIGRAMS, CREATE_NEW_ID_2_WORD, CREATE_NEW_CORPUS)
    texts = lemmas_bigrams
    # Human readable format of corpus (term-frequency)
    # [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

    #########Building LDA Mallet Model#########
    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    mallet_path = os.path.join(os.environ['MALLET_HOME'], 'bin','mallet')

    if greedy:
        greedy_num_of_topics_search(id2word, texts, corpus)
    else:
       current_lda_model = create_and_train_lda_model(mallet_path, corpus, id2word, number_of_topics, prefix, seed=RANDOM_SEED)

    turns_prefix = f'turns_{prefix}'
    EVALUATION_DATA_CREATE_NEW_LEMMAS = True
    EVALUATION_DATA_CREATE_NEW_LEMMA_BIGRAMS = True
    EVALUATION_DATA_CREATE_NEW_ID2WORD = False#should always be false so will use the same id2word
    EVALUATION_DATA_CREATE_NEW_CORPUS = True
    turns_path = "D:\\data\\gili\\processed_knesset\\protocols\\plenary_protocols\\plenary_protocols_jsons"
    #From here its classifying documents to the learned topics
    id2word_prefix = prefix
    turns_lemmas_bigrams, turns_id2word, turns_corpus, turns_text_ids = prepare_texts_for_lda([turns_path], turns_prefix, id2word_prefix,1, create_new_lemmas=EVALUATION_DATA_CREATE_NEW_LEMMAS, create_new_lemma_bigrams=EVALUATION_DATA_CREATE_NEW_LEMMA_BIGRAMS, create_new_id_2_word=EVALUATION_DATA_CREATE_NEW_ID2WORD, create_new_corpus=EVALUATION_DATA_CREATE_NEW_CORPUS, turns=True,text_ids=True)

    texts = turns_lemmas_bigrams
    data = texts
    corpus = turns_corpus
    text_ids = turns_text_ids

    now = datetime.now()
    current_time = now.strftime("%H:%M.:%S")
    print(f"started format_topics_sentences. current time: {current_time}", flush=True)
    filename = "plenary_turns_topics.csv"#
    df_topics = get_topics_for_documents(ldamodel=current_lda_model, corpus=turns_corpus, texts=data, text_ids=turns_text_ids, filename=filename, chunksize=100)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"finished format_topics_sentences.. current time: {current_time}", flush=True)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")





