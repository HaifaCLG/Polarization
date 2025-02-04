import os
FIRST_YEAR_MONTHS = ['01','02','03','04','05','06']
SECOND_YEAR_MONTHS = ['07','08','09','10','11','12']

WINTER_CHAIR_MONTHS = ["10", "11", "12", "01", "02", "03"]
WINTER_PERIOD_FIRST_YEAR_MONTHS = ["10", "11", "12"]
WINTER_PERIOD_SECOND_YEAR_MONTHS = ["01", "02", "03"]
SUMMER_PERIOD_MONTHS = ["04", "05", "06", "07", "08", "09"]


###########PATHS#################
data_path = "D:\\data\\gili"

##PROCESSED DATA PATHS##
processed_knesset_data_path = os.path.join(data_path,"processed_knesset")
knesset_protocols_path = os.path.join(processed_knesset_data_path, "protocols")
knesset_txt_files_path = os.path.join(processed_knesset_data_path, 'knesset_data_txt_files')
knesset_csv_files_path = os.path.join(processed_knesset_data_path, 'knesset_data_csv_files')
project_folder_path = "G:\\.shortcut-targets-by-id\\0B-jikcMdVWv0YXpLYUhFOTlGOFU\\OurDrive\\University of Haifa\\דוקטורט\\extremism"
committee_processed_protocols_path = os.path.join(knesset_protocols_path, "committee_protocols", "committee_protocols_jsons")
plenary_processed_protocols_path = os.path.join(knesset_protocols_path, "plenary_protocols", "plenary_protocols_jsons")
committee_full_sentences_shards_path = os.path.join(processed_knesset_data_path,"sentences_jsonl_files","committee_full_sentences_shards")
plenary_full_sentences_shards_path = os.path.join(processed_knesset_data_path,"sentences_jsonl_files","plenary_full_sentences_shards")
emobank_csv_path = os.path.join(project_folder_path, "vad_manual_annotations_hebrew","emobank.csv")
vad_manual_annotations_path = os.path.join(project_folder_path, "vad_manual_annotations_hebrew")
lemma_conllu_file_path = os.path.join(data_path, "noam_models","UD_Hebrew-IAHLTwiki","UD_Hebrew-IAHLTwiki","models","lexicon_words_lemmas.conllu")
enriched_final_lexicon_path = os.path.join(vad_manual_annotations_path, "enriched_final_lexicon.csv")