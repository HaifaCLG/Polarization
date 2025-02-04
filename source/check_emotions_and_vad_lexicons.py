import pandas as pd
from vad_functions import *

CONVERT_EMO_LEX_TO_CSV = False#should happen only once
def check_similarity_between_lexicons(word_lists_a, word_lists_b, name_lex_a=None, name_lex_b=None):

    only_in_a = [word for word in word_lists_a if word not in word_lists_b]
    only_in_b = [word for word  in word_lists_b if word not in word_lists_a]
    in_both_lexicons = [word for word in word_lists_a if word in word_lists_b]
    unique_in_both_lexicons = list(set(in_both_lexicons))
    big_list_word = word_lists_a+word_lists_b
    all_unique_words = set(big_list_word)
    print(f'total number of unique words: {len(all_unique_words)}')
    print(f'number of words in both lexicons: {len(in_both_lexicons)}')
    print(f'number of unique words in both lexicons: {len(unique_in_both_lexicons)}')

    if name_lex_a:
        lex_a_name = name_lex_a
    else:
        lex_a_name = "lex_a"
    print(f'number of words only in {lex_a_name} is {len(only_in_a)}')
    if name_lex_b:
        lex_b_name = name_lex_b
    else:
        lex_b_name = "lex_b"
    print(f'number of words only in {lex_b_name} is {len(only_in_b)}')

if __name__ == '__main__':
    csv_vad_lexicon_path = "vad_hebrew_english_original_lexicon.csv"
    all_vad_df = pd.read_csv(csv_vad_lexicon_path)
    all_vad_df = all_vad_df.dropna(how='any')
    vad_words_list = get_lexicon_words_from_df(all_vad_df, lang=Language.HEBREW)

    csv_emotion_lexicon_path = "emotions_hebrew_english_original_lexicon.csv"
    if CONVERT_EMO_LEX_TO_CSV:
        convert_lexicon_file_to_csv(HEBREW_ENGLISH_EMOTIONS_LEXICON_PATH, output_path=csv_emotion_lexicon_path)
    all_emo_df = pd.read_csv(csv_emotion_lexicon_path)
    all_emo_df = all_emo_df.dropna(how='any')
    emotions_words_list = get_lexicon_words_from_df(all_emo_df, lang=Language.HEBREW)
    check_similarity_between_lexicons(vad_words_list, emotions_words_list,"vad_lexicon","emotion_lexicon")

