import pandas as pd

def extract_annotator_row_index_by_name(df, name):
    res_row_index = -1
    for column_name, column_series in df.items():
        if column_name == "שם פרטי":
            for row_index, name_value in zip(range(len(column_series)),column_series):
                if str(name_value).strip() == name.strip():
                    res_row_index = row_index
                    break
    return res_row_index
def extract_three_annotators_row_index(df):
    shuly_row_index = 0
    ella_row_index = 0
    gili_row_index = 0
    shira_row_index = 0
    israel_row_index = 0
    avia_row_index = 0
    first_annotator = {}
    second_annotator = {}
    third_annotator = {}
    for column_name, column_series in df.items():
        if column_name == "שם פרטי":
            for row_index, name_value in zip(range(len(column_series)),column_series):
                if str(name_value).strip() == "שולי":
                    shuly_row_index = row_index
                elif str(name_value).strip() == "אלה":
                    ella_row_index = row_index
                elif str(name_value).strip() == "גילי":
                    gili_row_index = row_index
                elif str(name_value).strip() == "שירה":
                    shira_row_index = row_index
                elif str(name_value).strip() == "ישראל":
                    israel_row_index = row_index
                elif str(name_value).strip() == "אביה":
                    avia_row_index = row_index
                else:
                    continue
    if gili_row_index>0:
        first_annotator["index"] = shuly_row_index
        first_annotator["name"] = "shuly"
        second_annotator["index"] = ella_row_index
        second_annotator["name"] = "ella"
        third_annotator["index"] = gili_row_index
        third_annotator["name"] = "gili"
    else:
        first_annotator["index"] = shira_row_index
        first_annotator["name"] = "shira"
        second_annotator["index"] = israel_row_index
        second_annotator["name"] = "israel"
        third_annotator["index"] = avia_row_index
        third_annotator["name"] = "avia"
    return first_annotator,second_annotator,third_annotator


def get_annotators_scores_from_column(column_series, low_question, high_question, sent_number):
    if column_series[first_annotator_row_index] == "1":
        first_annotator_high_answer = sent_number
        high_question[first_annotator["name"]] = first_annotator_high_answer
    if column_series[first_annotator_row_index] == '2':
        first_annotator_low_answer = sent_number
        low_question[first_annotator["name"]] = first_annotator_low_answer
    if column_series[second_annotator_row_index] == "1":
        second_annotator_high_answer = sent_number
        high_question[second_annotator["name"]] = second_annotator_high_answer
    if column_series[second_annotator_row_index] == "2":
        second_annotator_low_answer = sent_number
        low_question[second_annotator["name"]] = second_annotator_low_answer
    if column_series[third_annotator_row_index] == "1":
        third_annotator_high_answer = sent_number
        high_question[third_annotator["name"]] = third_annotator_high_answer
    if column_series[third_annotator_row_index] == "2":
        third_annotator_low_answer = sent_number
        low_question[third_annotator["name"]] = third_annotator_low_answer

def get_answers(df, vad_value):
    vad_value_flag = False
    inside_question_sent_counter = 0
    high_questions = []
    low_questions = []
    for (column_name, column_series),column_index in zip(df.items(), range(len(df.columns))):
        if "בכל עמוד" in column_name:
            continue
        if vad_value in column_name:
            high_question = {}
            low_question = {}
            vad_value_flag = True
            inside_question_sent_counter +=1
            get_annotators_scores_from_column(column_series, low_question, high_question, inside_question_sent_counter)

        elif vad_value_flag and inside_question_sent_counter<4:
            inside_question_sent_counter +=1
            get_annotators_scores_from_column(column_series, low_question, high_question, inside_question_sent_counter)

            if inside_question_sent_counter>=4:
                inside_question_sent_counter = 0
                vad_value_flag = False
                high_questions.append(high_question)
                low_questions.append(low_question)

    return high_questions, low_questions


def get_question_annotation_agreement(question):
    first_answer_counter = 0
    second_answer_counter = 0
    third_answer_counter = 0
    fourth_answer_counter = 0
    for name, value in question.items():
        if value == 1:
            first_answer_counter +=1
        elif value == 2:
            second_answer_counter +=1
        elif value == 3:
            third_answer_counter +=1
        elif value == 4:
            fourth_answer_counter +=1
        else:
            print("WTF?")
    number_of_annotators = 3
    if first_answer_counter == number_of_annotators or second_answer_counter == number_of_annotators or third_answer_counter == number_of_annotators or fourth_answer_counter == number_of_annotators:
        score = 3#todo need to change if more than 3 annotators
    elif first_answer_counter == 2 or second_answer_counter == 2 or third_answer_counter == 2 or fourth_answer_counter == 2:
        score = 2
    else:
        score = 1

    return score


def check_agreement(questions_list, name_of_series):
    three_agreement_count = 0
    two_agreement_count = 0
    one_agreement_count = 0
    for question in questions_list:
        question_agreement_score = get_question_annotation_agreement(question)
        if question_agreement_score == 3:
            three_agreement_count +=1
        elif question_agreement_score == 2:
            two_agreement_count +=1
        elif question_agreement_score == 1:
            one_agreement_count +=1
        else:
            print("WTF?")
    print(f'Agreement scores for {name_of_series}: ')
    print(f'3-agreement: {three_agreement_count} (out of {len(questions_list)}), {"{0:0.3f}".format(100*(three_agreement_count/len(questions_list)))}%')
    print(f'2-agreement: {two_agreement_count} (out of {len(questions_list)}), {"{0:0.3f}".format(100*(two_agreement_count/len(questions_list)))}%')
    print(f'1-agreement: {one_agreement_count} (out of {len(questions_list)}), {"{0:0.3f}".format(100*(one_agreement_count/len(questions_list)))}%\n')


if __name__ == '__main__':
    # df = pd.read_csv("shuly-ella-gili-annotations-1.csv")
    # df = pd.read_csv("shuly-ella-gili-annotations-2.csv")
    df = pd.read_csv("shira-israel-avia-annotations-1.csv")


    first_annotator, second_annotator, third_annotator = extract_three_annotators_row_index(df)
    first_annotator_row_index = first_annotator["index"]
    second_annotator_row_index = second_annotator["index"]
    third_annotator_row_index = third_annotator["index"]
    valence_high_questions, valence_low_questions = get_answers(df, "Valence")
    arousal_high_questions, arousal_low_questions = get_answers(df, "Arousal")
    dominance_high_questions, dominance_low_questions = get_answers(df, "Dominance")
    all_questions = valence_high_questions + valence_low_questions + arousal_high_questions + arousal_low_questions + dominance_high_questions+ dominance_low_questions
    all_high_value_questions = valence_high_questions + arousal_high_questions + dominance_high_questions
    all_low_value_questions = valence_low_questions + arousal_low_questions + dominance_low_questions
    valence_questions = valence_high_questions + valence_low_questions
    arousal_questions = arousal_high_questions + arousal_low_questions
    dominance_questions = dominance_high_questions + dominance_low_questions
    check_agreement(valence_high_questions, "valence_high_questions")
    check_agreement(valence_low_questions, "valence_low_questions")
    check_agreement(arousal_high_questions, "arousal_high_questions")
    check_agreement(arousal_low_questions, "arousal_low_questions")
    check_agreement(dominance_high_questions, "dominance_high_questions")
    check_agreement(dominance_low_questions, "dominance_low_questions")
    check_agreement(all_questions, "all_questions")
    check_agreement(all_high_value_questions, "all_high_value_questions")
    check_agreement(all_low_value_questions, "all_low_value_questions")
    check_agreement(valence_questions, "valence_questions")
    check_agreement(arousal_questions, "arousal_questions")
    check_agreement(dominance_questions, "dominance_questions")
