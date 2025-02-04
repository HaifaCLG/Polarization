import pandas as pd

from vad_sentences_annotations import extract_annotator_row_index_by_name


def get_english_name_from_hebrew_name(annotator_name):
    if annotator_name.strip() == "שירה":
        return "shira"
    elif annotator_name.strip() == "ישראל":
        return "israel"
    elif annotator_name.strip() == "אביה":
        return "avia"
    else:
        print(f'wrong hebrew name: {annotator_name}')


if __name__ == '__main__':
    sent_num_tuples_file = "sent_numbers_tuples_comma_separated.txt"
    annotations_files = ["qp_annotations_1.csv","qp_annotations_2.csv", "qp_annotations_3.csv", "qp_annotations_4.csv", "qp_annotations_5.csv", "qp_annotations_6.csv", "qp_annotations_7.csv", "qp_annotations_8.csv", "qp_annotations_9.csv", "qp_annotations_10.csv"]
    annotator_name = "אביה"
    VAD_feature = "Valence"
    tuples_annotations = []
    sent_tuple_counter = -1
    with open(sent_num_tuples_file) as file:
        sent_num_tuples_list = file.readlines()
    for annotation_file in annotations_files:
        annotation_df = pd.read_csv(annotation_file)
        INSIDE_FEATURE_QUESTION = False
        row_index = extract_annotator_row_index_by_name(annotation_df, annotator_name)
        if row_index < 0:
            print("error: wrong row index")
            exit()
        for column_name, column_series in annotation_df.items():
            if VAD_feature in column_name and "דפים" not in column_name:
                inside_question_counter = 0
                best = -1
                worst = -1
                sent_tuple_counter += 1
                tuple_str = sent_num_tuples_list[sent_tuple_counter]
                tuple = tuple_str.split(",")
                INSIDE_FEATURE_QUESTION = True
            if INSIDE_FEATURE_QUESTION:
                val = column_series[row_index]
                if val == '1':
                    best = tuple[inside_question_counter].strip()
                elif val == '2':
                    worst = tuple[inside_question_counter].strip()
                else:
                    inside_question_counter += 1
                    continue

                if best != -1 and worst != -1:
                    graded_tuple_row = []
                    for i in range(4):
                        graded_tuple_row.append(tuple[i].strip())
                    graded_tuple_row.append(best)
                    graded_tuple_row.append(worst)
                    tuples_annotations.append(graded_tuple_row)
                    INSIDE_FEATURE_QUESTION = False
                inside_question_counter += 1
    english_annotator_name = get_english_name_from_hebrew_name(annotator_name)
    res_df = pd.DataFrame(tuples_annotations, columns=["Item1", "Item2", "Item3", "Item4", "BestItem", "WorstItem"])
    res_df.to_csv(f'{english_annotator_name}_{VAD_feature}_tuples_annotations.csv', index=False)


