from numpy.linalg import norm

from vad_functions import *
from sentence_transformers.util import cos_sim
from scipy.stats import pearsonr, spearmanr, kendalltau
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import pandas as pd
import random
import pandas as pd
random.seed(42)

def predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer = None, vad_column="V", print=False):
    pretty = False
    if ALEPHBERTGIMMEL:
        if tokenizer:
            sentences_embeddings = get_model_sentences_embeddings_from_last_hidden_state(sentences, sent_embeddings_model, tokenizer)
        else:
            print("no tokenizer was passed for alephbertgimmel model")
            exit()
    elif DICTABERT or OUR_DICTA_PRETRAINED_MODEL:
        if tokenizer:
            sentences_embeddings = get_model_sentences_embeddings_from_last_hidden_state(sentences, sent_embeddings_model, tokenizer)
        else:
            print("no tokenizer was passed for model")
            exit()
    elif YAM_PELEG:
        if tokenizer:
            sentences_embeddings = get_gpt_style_sentence_embeddings_from_last_hidden_state(sentences, sent_embeddings_model, tokenizer)
        else:
            print("no tokenizer was passed for model")
            exit()
    else:
        sentences_embeddings = sent_embeddings_model.encode(sentences)
    if vad_column == "V":
        column_name = "Valence"
        binom_model = load_object(Valence_regression_model_name)
    elif vad_column == "A":
        binom_model = load_object(Arousal_regression_model_name)
        column_name = "Arousal"
    elif vad_column == "D":
        binom_model = load_object(Dominance_regression_model_name)
        column_name = "Dominance"
    else:
        print(f'wrong vad column: {vad_column}')

    preds = binom_model.predict(sentences_embeddings)
    if print:
        if pretty:
            pretty_print_preds_with_sentences(column_name, preds, sentences)
        else:
            print_sentences_scores(column_name, preds, sentences)

    return preds

def print_sentences_scores(column_name, preds, sentences):
    print(f'The values of {column_name} are:')
    with_text = False
    for sent_index,sent, val in zip(range(1,len(sentences)+1),sentences, preds):
        if with_text:
            print(f'{sent.strip()}\t{sent_index}\t{val}')
        else:
            print(f'{sent_index}\t{val}')

    print('\n')
def pretty_print_preds_with_sentences(column_name, preds, sentences):
    formatted_preds = ['%.2f' % elem for elem in preds]
    print(f'The values of {column_name} are:')
    for sent, val in zip(sentences, formatted_preds):
        print(f'{sent.strip()}: value is: {val}')
    print('\n')


def predict_vad_values_on_high_and_low_grade_sentences(emo_bank_df, bert_sent_embeddings_model,V=True, A=True, D=True, num_of_sentences = 5, high=True, low=False):
    if V:
        if high:
            v_high_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "V", num_of_sentences)
            predict_one_vad_value_on_sentences(v_high_grade_sentences, bert_sent_embeddings_model, vad_column="V")
        if low:
            v_low_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "V", num_of_sentences, high=False)
            predict_one_vad_value_on_sentences(v_low_grade_sentences, bert_sent_embeddings_model, vad_column="V")
    if A:
        if high:
            a_high_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "A", num_of_sentences)
            predict_one_vad_value_on_sentences(a_high_grade_sentences, bert_sent_embeddings_model, vad_column="A")
        if low:
            a_low_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "A", num_of_sentences, high=False)
            predict_one_vad_value_on_sentences(a_low_grade_sentences, bert_sent_embeddings_model, vad_column="A")
    if D:
        if high:
            d_high_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "D", num_of_sentences)
            predict_one_vad_value_on_sentences(d_high_grade_sentences, bert_sent_embeddings_model, vad_column="D")
        if low:
            d_low_grade_sentences = get_high_or_low_grade_sentences(emo_bank_df, "D", num_of_sentences, high=False)
            predict_one_vad_value_on_sentences(d_low_grade_sentences, bert_sent_embeddings_model, vad_column="D")

def predict_vad_values_on_costume_sentences(sent_embeddings_model, tokenizer=None, hebrew=False, multilang_model=False):
    sentences = []
    valance_sentences = []
    arousal_sentences = []
    dominance_sentences = []
    Knesset = True
    if Knesset:
        valance_sentences.append("זה ממש עצוב לי ששם אנחנו נמצאים.")
        valance_sentences.append("מאוד מעציב אותי שככה הכנסת מתנהלת.")
        valance_sentences.append("אני חושב שזה הדבר הכי משמח, ואני גם רוצה להגיד למה.")
        valance_sentences.append("אני רוצה לעבור לנושא שהוא יותר משמח.")
        valance_sentences.append("נהדר, אני מאוד שמחה.")
        valance_sentences.append("אנחנו כולנו בתקופה חגיגת של ערב המשחקים האולימפיים והמשחקים הפרא-אולימפיים – מצפה לנו קיץ מהנה ומרתק.")
        valance_sentences.append("אמר פה גדעון שלום, וזה נכון, שמשרד האוצר נדיב מאי פעם.")
        valance_sentences.append("זה סיוט שאנשים עוברים.")
        valance_sentences.append("אבל מה שקורה היום בתחום הדיור כמעט מייאש.")
        arousal_sentences.append("ואפילו תקפו אותנו שאנחנו מתנגדים לתת רישיונות ייבוא.")
        arousal_sentences.append("כשהגיעו לשם שוטרים הם תקפו את השוטרים.")
        arousal_sentences.append("כל הקונספט של הטיפול הוא כושל, מי מכניס בכפייה בן אדם לטיפול אלים?")
        arousal_sentences.append("וכל זמן שהעומס הזה יהיה, תהיה אלימות.")
        arousal_sentences.append("לא רק טרור פיזי והורגים אותנו בגרזנים.")
        arousal_sentences.append("והיו שני מקרי רצח מזעזעים, אמרתי, גם בעכו וגם כפר קרע.")
        arousal_sentences.append("הוא רגיל לישון צוהריים, בנט.")
        arousal_sentences.append("הוא הלך לישון.")
        arousal_sentences.append("אני הולך לנוח במלון.")
        arousal_sentences.append("קצת מנמנם אחרי יום עבודה קשה.")
        dominance_sentences.append("אני מברך על ההחלטה של מזרח ירושלים - צעד אמיץ, ראוי, משמעותי.")
        dominance_sentences.append("שר האוצר יצא גדול, יצא אמיץ.")
        dominance_sentences.append("עכשיו אני כבר לא אומר את זה כי אני יודע שזה חסר תכלית.")
        dominance_sentences.append("אני חלש במתמטיקה.")
        dominance_sentences.append("עכשיו לצפות ממנו, כשהוא היום מאוד חלש.")
        dominance_sentences.append("ידוע לו, אם הוא לא יודע, הוא לא יודע.")
        dominance_sentences.append("אני לא יודע איך לצאת מהמצב הזה.")
        dominance_sentences.append("אני בטוח שהוא יכול לשלוח.")
        dominance_sentences.append("הייתה ממשלה של ימין חזק.")
        dominance_sentences.append("כל אחד ואחת יכול להפגין מנהיגות ולהשפיע על העולם היהודי.")
        dominance_sentences.append("היא הוכיחה מנהיגות, הובלה וראיית הנולד.")

    else:
        if hebrew:
            sent_1 = "אני כל כך שמחה לפגוש אותך"
        else:
            sent_1 = "I am so happy to see you"
        valance_sentences.append(sent_1)
        if hebrew:
            sent_2 = "זה ממש מעציב אותי"
        else:
            sent_2 = "I am really sad about this"
        valance_sentences.append(sent_2)
        if hebrew:
            sent_3 = "הם תקפו אותו באלימות"
        else:
            sent_3 = "they attacked him violently"
        arousal_sentences.append(sent_3)
        if hebrew:
            sent_4 = "אני הולך לישון"
        else:
            sent_4 = "I am going to sleep"
        arousal_sentences.append(sent_4)
        if hebrew:
            sent_5 = "הוא חזק ואמיץ"
        else:
            sent_5 = "he is strong and brave"
        dominance_sentences.append(sent_5)
        if hebrew:
            sent_6 = "אני לא יודע, מה שתגיד"
        else:
            sent_6 = "I don't know, what ever you say"
        dominance_sentences.append(sent_6)
        if hebrew:
            sent_7="אני מרגיש חלש וחסר תכלית"
        else:
            sent_7 = "I feel weak and pointless"
        dominance_sentences.append(sent_7)
    if multilang_model:
        valance_sentences = process_input_texts_for_multi_model(valance_sentences)
        arousal_sentences = process_input_texts_for_multi_model(arousal_sentences)
        dominance_sentences = process_input_texts_for_multi_model(dominance_sentences)
    predict_one_vad_value_on_sentences(valance_sentences, sent_embeddings_model,tokenizer=tokenizer, vad_column="V", print=True)
    predict_one_vad_value_on_sentences(arousal_sentences, sent_embeddings_model, tokenizer=tokenizer,vad_column="A",print=True)
    predict_one_vad_value_on_sentences(dominance_sentences, sent_embeddings_model, tokenizer=tokenizer,vad_column="D",print=True)




#a = (a-min)/(max-min)
def get_scaled_columns_values(df,column_name):
    scaled = (df[column_name] - df[column_name].min()) / (
            df[column_name].max() - df[column_name].min())
    return scaled.values

def get_high_or_low_grade_sentences(df, column_name,num_of_sentences, high=True):
    if high:
        sorted_df = df.sort_values(by=column_name, ascending=False)
    else:
        sorted_df = df.sort_values(by=column_name, ascending=True)

    sentences = list(sorted_df["text"].values)[:num_of_sentences]
    return sentences


def evaluate_vad_predictions_on_emo_bank_sents(eb_df, bert_sent_embeddings_model):
    v_binom_model = load_object(Valence_regression_model_name)
    a_binom_model = load_object(Arousal_regression_model_name)
    d_binom_model = load_object(Dominance_regression_model_name)
    sampled_eb_sentences_df = eb_df.sample(n=1000, random_state=0)
    sampled_eb_sentences_texts = list(sampled_eb_sentences_df["text"].values)
    # v_y = get_scaled_columns_values(sampled_eb_sentences_df, "V")
    # a_y = get_scaled_columns_values(sampled_eb_sentences_df, "A")
    # d_y = get_scaled_columns_values(sampled_eb_sentences_df, "D")
    #
    v_y = sampled_eb_sentences_df["V"].values
    a_y = sampled_eb_sentences_df["A"].values
    d_y = sampled_eb_sentences_df["D"].values
    sampled_eb_sentences_embeddings = bert_sent_embeddings_model.encode(sampled_eb_sentences_texts)
    evaluate_model(v_binom_model, sampled_eb_sentences_embeddings, v_y, "Valence emoBank sentences")
    evaluate_model(a_binom_model, sampled_eb_sentences_embeddings, a_y, "Arousal emoBank sentences")
    evaluate_model(d_binom_model, sampled_eb_sentences_embeddings, d_y, "Dominance emoBank sentences")


def check_similarity_of_embeddings_vectors(sent_embeddings_model, sent_1, sent_2, tokenizer=None):
    if ALEPHBERTGIMMEL:
        if tokenizer:
            sentences_embeddings = get_model_sentences_embeddings_from_last_hidden_state([sent_1, sent_2], sent_embeddings_model, tokenizer)
        else:
            print("no tokenizer was passes for alephbertgimmel model")
            exit()
    elif DICTABERT or OUR_DICTA_PRETRAINED_MODEL:
        if tokenizer:
            sentences_embeddings = get_model_sentences_embeddings_from_last_hidden_state([sent_1, sent_2],
                                                                                         sent_embeddings_model,
                                                                                         tokenizer)
        else:
            print("no tokenizer was passes for model")
            exit()
    else:
        sentences_embeddings = sent_embeddings_model.encode([sent_1, sent_2])
    sent_1_emb = sentences_embeddings[0]
    sent_2_emb = sentences_embeddings[1]
    cosine = np.dot(sent_1_emb, sent_2_emb) / (norm(sent_1_emb) * norm(sent_2_emb))
    print(f"sent 1 is: {sent_1}")
    print(f"sent 2 is: {sent_2}")
    print("Cosine Similarity:", cosine)
    #
    # print("other option:")
    # print(cos_sim(*tuple(sentences_embeddings)).item())



def predict_vad_values_on_emobank(sent_embeddings_model, tokenizer, vad_column="V"):
    # print(kendalltaualization.load_obj('binom.model.' + 'v')
    sentences, emobank_values = select_and_prepare_emobank_sentences(vad_column)


    if MULTI or OUR_MULTI_PRETRAINED_MODEL:
        sentences = process_input_texts_for_multi_model(sentences)
    # predictions = infer_emotion_value(model.encode(sentences), regressor)  # predict dimension score
    predictions = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer, vad_column=vad_column,
                                       print=False)
    print(f'pearsonr: {pearsonr(emobank_values, predictions)}')
    print(f'spearmanr: {spearmanr(emobank_values, predictions)}')
    print(f'kendalltau: {kendalltau(emobank_values, predictions)}')

def check_similiarity_on_different_sentences(sent_embeddings_model, tokenizer):
    Knesset = True
    if Knesset:
        sent_1 = "התחלואה הכפולה, מתמודדים עם תחלואה כפולה, עם תפוח אדמה לוהט."
        sent_2 = "זה תפוח אדמה לוהט שכל שר גלגל אותו ליד השנייה של השר הזה, וצריך לומר באומץ."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "כשנכנסתי לכאן כחברת כנסת, לוועדת המשנה למוכנות העורף, הייתה לי בטן מלאה מניסיון העבר שלי."
        sent_2 = "כשמוניתי להיות חברת כנסת, היה לי ברור דבר אחד – המשימה שלי היא מוכנות העורף."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "זה היה בתקנות תו ירוק עובדים."
        sent_2 = "לא, אתם ביטלתם כרגע את תקנות תו ירוק לעובדים."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "אם אתם מדברים על העניין של פרטיות, השאלה אם אוזניות לאדם כבד שמיעה שבא לבנק ואומר שהוא לא שומע, לתת לו אוזניות."
        sent_2 = "הם ידעו שזה מחכה להם בבנק, כי זה יהיה מונגש וזה יהיה בכל מקום והם יביאו אוזניות משלהם."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "ארץ ישראל שייכת לעם היהודי."
        sent_2 = "ארץ ישראל, מדינת היהודים היא מדינת הלאום של העם היהודי."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "ארץ ישראל שייכת לעם היהודי."
        sent_2 = "התחלואה הכפולה, מתמודדים עם תחלואה כפולה, עם תפוח אדמה לוהט."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "הם ידעו שזה מחכה להם בבנק, כי זה יהיה מונגש וזה יהיה בכל מקום והם יביאו אוזניות משלהם."
        sent_2 = "תקשיבו, חברים, אני מרימה דגל אדום."
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)
    else:
        sent_1 = "ארץ ישראל היפה"
        sent_2 = "טוש ירוק ללוח"
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "אני אוהב אבטיח"
        sent_2 = "אני אוהבת מלון"
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "כובע"
        sent_2 = "קסדה"
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "שחור"
        sent_2 = "לבן"
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

        sent_1 = "פצע"
        sent_2 = "יפה"
        if MULTI or OUR_MULTI_PRETRAINED_MODEL:
            sents = process_input_texts_for_multi_model([sent_1, sent_2])
        else:
            sents = [sent_1, sent_2]
        check_similarity_of_embeddings_vectors(sent_embeddings_model, sents[0], sents[1], tokenizer=tokenizer)

def try_english_models_on_europarl_and_emobank(sent_embeddings_model):
    if LANGUAGE == Language.ENGLISH:
        RESAMPLE_EUROPARL_SENTS = True
        REPREDICT_EUROPARL_VAD_VALUES = True
        k = 2000
        if RESAMPLE_EUROPARL_SENTS:
            english_europarl_sentences_path = "C:\\Users\\gilis\\Downloads\\de-en\\europarl-v7.de-en.en"
            with open(english_europarl_sentences_path, encoding="utf-8") as file:
                english_europarl_sentences = file.readlines()
            english_europarl_sentences = list(set(english_europarl_sentences))
            english_europarl_sentences = [x for x in english_europarl_sentences if x.strip() != '']
            sampled_english_eurparl_sents = random.sample(english_europarl_sentences, k=k)
            if MULTI:
                sampled_english_eurparl_sents = process_input_texts_for_multi_model(sampled_english_eurparl_sents)

            save_object(sampled_english_eurparl_sents, f"sampled_english_eurparl_sents_{k}_with_query")
        else:
            sampled_english_eurparl_sents = load_object(f"sampled_english_eurparl_sents_{k}_with_query")
        if REPREDICT_EUROPARL_VAD_VALUES:
            v_preds = predict_one_vad_value_on_sentences(sampled_english_eurparl_sents, sent_embeddings_model,
                                                         vad_column="V")
            save_object(v_preds, f"multi_europarl_v_preds_{k}")
            a_preds = predict_one_vad_value_on_sentences(sampled_english_eurparl_sents, sent_embeddings_model,
                                                         vad_column="A")
            save_object(a_preds, f"multi_europarl_a_preds_{k}")
            d_preds = predict_one_vad_value_on_sentences(sampled_english_eurparl_sents, sent_embeddings_model,
                                                         vad_column="D")
            save_object(d_preds, f"multi_europarl_d_preds_{k}")
        else:
            v_preds = load_object(f"multi_europarl_v_preds_{k}")
            a_preds = load_object(f"multi_europarl_a_preds_{k}")
            d_preds = load_object(f"multi_europarl_d_preds_{k}")

        num_of_highest_and_lowest_sentences = 5
        n = num_of_highest_and_lowest_sentences
        sorted_by_v_europarl_sentences = [x for _, x in
                                          sorted(zip(v_preds, sampled_english_eurparl_sents), reverse=True)]
        sorted_v_preds = list(v_preds)
        sorted_v_preds.sort(reverse=True)
        print(f'europarl sentences with highest V:')
        pretty_print_preds_with_sentences("V", sorted_v_preds[:n], sorted_by_v_europarl_sentences[:n])
        print(f'europarl sentences with lowest V:')  # todo restore
        pretty_print_preds_with_sentences("V", sorted_v_preds[-n:], sorted_by_v_europarl_sentences[-n:])

        sorted_by_a_europarl_sentences = [x for _, x in
                                          sorted(zip(a_preds, sampled_english_eurparl_sents), reverse=True)]
        sorted_a_preds = list(a_preds)
        sorted_a_preds.sort(reverse=True)
        print(f'europarl sentences with highest A:')
        pretty_print_preds_with_sentences("A", sorted_a_preds[:n], sorted_by_a_europarl_sentences[:n])
        print(f'europarl sentences with lowest A:')
        pretty_print_preds_with_sentences("A", sorted_a_preds[-n:], sorted_by_a_europarl_sentences[-n:])

        sorted_by_d_europarl_sentences = [x for _, x in
                                          sorted(zip(d_preds, sampled_english_eurparl_sents), reverse=True)]
        sorted_d_preds = list(d_preds)
        sorted_d_preds.sort(reverse=True)
        print(f'europarl sentences with highest D:')
        pretty_print_preds_with_sentences("D", sorted_d_preds[:n], sorted_by_d_europarl_sentences[:n])
        print(f'europarl sentences with lowest D:')
        pretty_print_preds_with_sentences("D", sorted_d_preds[-n:], sorted_by_d_europarl_sentences[-n:])

        # eb_df = pd.read_csv("emobank.csv", index_col=0)

        # evaluate_vad_predictions_on_emo_bank_sents(eb_df, bert_sent_embeddings_model)

        # predict_vad_values_on_high_and_low_grade_sentences(eb_df, bert_sent_embeddings_model, V=True, A=True, D=True, num_of_sentences=5, high=True, low=True)


if __name__ == '__main__':
    if ALEPHBERTGIMMEL:
        sent_embeddings_model, tokenizer = get_alephbertgimmel_sentence_encoder_model()
    elif DICTABERT:
        sent_embeddings_model, tokenizer = get_dictabert_sentence_encoder_model()
    elif OUR_DICTA_PRETRAINED_MODEL:
        model_output_dir = "fine_tuned_on_dlc_server//model"
        sent_embeddings_model, tokenizer = get_auto_model_and_tokenizer(model_output_dir)
    elif OUR_MULTI_PRETRAINED_MODEL:
        fine_tuned_model_reassembled = load_our_fine_tuned_multi_model()
        sent_embeddings_model = fine_tuned_model_reassembled
        tokenizer = None

    else:
        sent_embeddings_model = SentenceTransformer(SENT_TRANSFORMER_TYPE)
        tokenizer = None


    all_knesset_sentences, all_knesset_sentences_vad_values_df = select_knesset_sentences(1)
    if TRAIN_WITH_KNESSET_DATA:

        trained_knesset_sentences = load_object(f'knesset_training_sentences')
        test_sentences = [sent for sent in all_knesset_sentences if sent not in trained_knesset_sentences]
        sentences = test_sentences
        test_df = all_knesset_sentences_vad_values_df[all_knesset_sentences_vad_values_df['sent_text'].isin(test_sentences)]
        ordered_test_df = test_df.set_index('sent_text').loc[test_sentences].reset_index()
        ordered_sentences = ordered_test_df['sent_text'].tolist()
        v_ordered_values = ordered_test_df['v-normalized-score'].tolist()
        a_ordered_values = ordered_test_df['a-normalized-score'].tolist()
        d_ordered_values = ordered_test_df['d-normalized-score'].tolist()

        print(f'test knesset sentences:\n {ordered_sentences}')
        print(f'v-knesset-test-real-values:\n {v_ordered_values}')
    else:
        test_sentences = all_knesset_sentences
        sentences = test_sentences
        v_ordered_values = all_knesset_sentences_vad_values_df['v-normalized-score'].tolist()
        a_ordered_values = all_knesset_sentences_vad_values_df['a-normalized-score'].tolist()
        d_ordered_values = all_knesset_sentences_vad_values_df['d-normalized-score'].tolist()
        print(f'knesset sentences:\n {all_knesset_sentences}')
        print(f'v-knesset-test-real-values:\n {v_ordered_values}')
    if MULTI or OUR_MULTI_PRETRAINED_MODEL:
        sentences = process_input_texts_for_multi_model(test_sentences)
    v_preds = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer, vad_column="V",
                                       print=True)
    res = pearsonr(v_preds, v_ordered_values)
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation on avg annotators test knesset sentences scores and multi-score trained on knesset and emo: statistic: {'%.3f' % statistic}, pvalue: {'%.3f' % pvalue}")

    print(f'a-knesset-test-real-values:\n {a_ordered_values}')

    a_preds = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer, vad_column="A",
                                       print=True)

    print(f'a-knesset-test-real-values:\n {d_ordered_values}')
    res = pearsonr(a_preds, a_ordered_values)
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation on avg annotators test knesset sentences scores and multi-score trained on knesset and emo: statistic: {'%.3f' % statistic}, pvalue: {'%.3f' % pvalue}")

    print(f'd-knesset-test-real-values:\n {a_ordered_values}')
    d_preds = predict_one_vad_value_on_sentences(sentences, sent_embeddings_model, tokenizer=tokenizer, vad_column="D",
                                       print=True)

    res = pearsonr(d_preds, d_ordered_values)
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation on avg annotators test knesset sentences scores and multi-score trained on knesset and emo: statistic: {'%.3f' % statistic}, pvalue: {'%.3f' % pvalue}")


    # predict_vad_values_on_costume_sentences(sent_embeddings_model, tokenizer=tokenizer, hebrew=True, multilang_model=MULTI or OUR_MULTI_PRETRAINED_MODEL)



