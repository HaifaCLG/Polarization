import os

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as sm_formula
from statsmodels.formula.api import ols
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing

from params_and_config import processed_knesset_data_path

def create_OLS_model_for_VAD_and_other_featues_for_session(session_stats_path, male_female_number_path, coalition_opposition_number_path, predicted_score_field_name="protocol_a_avg"):
    df_protocols_genders = pd.read_csv(male_female_number_path)
    eliminate_protocols_with_no_identified_knesset_members(df_protocols_genders, "num_female_members", "num_male_members")
    df_protocols_genders.sort_values(by='protocol_name', inplace=True)
    df_protocols_genders['proportion_females'] = df_protocols_genders['num_female_members'] / (df_protocols_genders['num_male_members'] + df_protocols_genders['num_female_members'])

    df_protocols_coalition_opposition = pd.read_csv(coalition_opposition_number_path)
    eliminate_protocols_with_no_identified_knesset_members(df_protocols_coalition_opposition, "num_coalition_members", "num_opposition_members")
    df_protocols_coalition_opposition.sort_values(by='protocol_name', inplace=True)
    df_protocols_coalition_opposition['proportion_coalition'] = df_protocols_coalition_opposition['num_coalition_members']/(df_protocols_coalition_opposition['num_coalition_members']+df_protocols_coalition_opposition['num_opposition_members'])

    df_vad_scores = pd.read_csv(session_stats_path)
    df_vad_scores.sort_values(by='protocol_name', inplace=True)

    df_merged = pd.merge(df_protocols_genders, df_vad_scores, on='protocol_name', sort=False)
    df_merged = pd.merge(df_protocols_coalition_opposition, df_merged, on='protocol_name', sort=False)



    # Add time points as features (assuming the protocols are sorted by date)
    df_merged['time_point'] = range(1, len(df_merged) + 1)
    check_multicollinearity(df_merged)
    X = df_merged[['proportion_females', 'proportion_coalition', 'time_point']]
    y = df_merged[predicted_score_field_name]
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    X = sm.add_constant(X_scaled_df)  # Adds a constant term to the predictor

    # X_time_only = df_merged[['time_point']]
    # model_time_only = sm.OLS(y, X_time_only).fit()
    # print(model_time_only.summary())
    model = sm.OLS(y, X).fit()

    # Print the summary
    print(model.summary())

    # Calculate Pearson correlation coefficients and p-values
    features = ['proportion_females', 'proportion_coalition', 'time_point']
    correlation_results = {}
    for feature in features:
        corr_coef, p_value = pearsonr(df_merged[feature], df_merged[predicted_score_field_name])
        correlation_results[feature] = (corr_coef, p_value)

    # Print Pearson correlation results
    for feature, (corr_coef, p_value) in correlation_results.items():
        print(f"Pearson correlation between {predicted_score_field_name} and {feature}:")
        print(f"  Correlation coefficient: {corr_coef}")
        print(f"  P-value: {p_value}\n")
def eliminate_protocols_with_no_identified_knesset_members(df, first_field_name, second_field_name):
    indices_to_drop = []

    for index, row in df.iterrows():
        if row[first_field_name] == 0 and row[second_field_name] == 0:
            indices_to_drop.append(index)

    # Drop the collected indices
    df.drop(indices_to_drop, inplace=True)

def create_committee_merged_df_with_female_and_coalition_proportion(session_stats_path, male_female_number_path, coalition_opposition_number_path, session_name):
    df_protocols_genders = pd.read_csv(male_female_number_path)
    eliminate_protocols_with_no_identified_knesset_members(df_protocols_genders, "num_female_members", "num_male_members")
    df_protocols_genders.sort_values(by='protocol_name', inplace=True)
    df_protocols_genders['proportion_females'] = df_protocols_genders['num_female_members'] / (df_protocols_genders['num_male_members'] + df_protocols_genders['num_female_members'])

    df_protocols_coalition_opposition = pd.read_csv(coalition_opposition_number_path)
    eliminate_protocols_with_no_identified_knesset_members(df_protocols_coalition_opposition, "num_coalition_members", "num_opposition_members")
    df_protocols_coalition_opposition.sort_values(by='protocol_name', inplace=True)
    df_protocols_coalition_opposition['proportion_coalition'] = df_protocols_coalition_opposition['num_coalition_members']/(df_protocols_coalition_opposition['num_coalition_members']+df_protocols_coalition_opposition['num_opposition_members'])

    df_vad_scores = pd.read_csv(session_stats_path)
    df_vad_scores.sort_values(by='protocol_name', inplace=True)

    df_merged = pd.merge(df_protocols_genders, df_vad_scores, on='protocol_name', sort=False)
    df_merged = pd.merge(df_protocols_coalition_opposition, df_merged, on='protocol_name', sort=False)

    committee_name = session_name
    df_merged['committee_name'] = committee_name
    return df_merged

def create_OLS_model_for_VAD_and_other_features(data_df, predicted_score_field_name="protocol_a_avg"):
    data_df.sort_values(by='protocol_name', inplace=True)
    # Add time points as features (assuming the protocols are sorted by date)
    data_df['time_point'] = range(1, len(data_df) + 1)
    num_cols = ['proportion_females', 'proportion_coalition', 'time_point']
    data_df[num_cols] = preprocessing.StandardScaler().fit_transform(X=data_df[num_cols])
    X = data_df
    # regression_formula = f'{predicted_score_field_name} ~ committee_name + committee_name*time_point + committee_name*proportion_females  + committee_name*proportion_coalition'
    regression_formula = f'{predicted_score_field_name} ~ committee_name + committee_name:time_point + committee_name:proportion_females  + committee_name:proportion_coalition -1'


    model = ols(formula=regression_formula, data=X)
    res = model.fit()
    print(res.summary())
    results = res.summary()
    res_df = pd.DataFrame(results.tables[1])
    base_line_committee = "work_welfare_health"
    # res_df.to_csv(f'ols_summery_{predicted_score_field_name}_{base_line_committee}.csv', index=False)
    res_df.to_csv(f'ols_summery_{predicted_score_field_name}.csv', index=False)
    df = res.wald_test_terms().summary_frame()
    df.to_csv(f"wald_test_terms_{predicted_score_field_name}.csv")#TODO restore
    # df.to_csv(f"wald_test_terms_{predicted_score_field_name}_{base_line_committee}.csv")#TODO remove, only for debug


    # Calculate Pearson correlation coefficients and p-values
    # features = ['proportion_females', 'proportion_coalition', 'time_point']
    # correlation_results = {}
    # for feature in features:
    #     corr_coef, p_value = pearsonr(data_df[feature], data_df[predicted_score_field_name])
    #     correlation_results[feature] = (corr_coef, p_value)
    #
    # # Print Pearson correlation results
    # for feature, (corr_coef, p_value) in correlation_results.items():
    #     print(f"Pearson correlation between {predicted_score_field_name} and {feature}:")
    #     print(f"  Correlation coefficient: {corr_coef}")
    #     print(f"  P-value: {p_value}\n")
def check_multicollinearity(df_merged):
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df_merged[['proportion_females', 'proportion_coalition', 'time_point']]
    X = sm.add_constant(X)

    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif)

if __name__ == '__main__':
    vad_values_committees_dir = os.path.join(processed_knesset_data_path,
                                             "sentences_jsonl_files\\sentences_with_vad_values\\committees")
    sessions_english_names = os.listdir(vad_values_committees_dir)
    sessions_english_names.remove("plenary")
    #removing sessions with less than 400K sentences
    sessions_english_names.remove("drugs")
    sessions_english_names.remove("foreign_workers")
    sessions_english_names.remove("security_and_out")

    committees_dfs = []
    for session_english_name in sessions_english_names:
        this_session_dir = os.path.join(vad_values_committees_dir, session_english_name)
        vad_output_dir_path = os.path.join(this_session_dir, "vad_shards")

        coalition_opposition_number_path = os.path.join(this_session_dir,
                                                        f"{session_english_name}_coalition_opposition_number.csv")
        male_female_number_path = os.path.join(this_session_dir, f"{session_english_name}_male_female_number.csv")
        session_stats_path = os.path.join(this_session_dir, f"{session_english_name}_vad_stats.csv")
        session_df = create_committee_merged_df_with_female_and_coalition_proportion(session_stats_path, male_female_number_path, coalition_opposition_number_path, session_english_name)
        committees_dfs.append(session_df)
    all_committees_merged_df = pd.concat(committees_dfs, ignore_index=True)
    # create_OLS_model_for_VAD_and_other_features(all_committees_merged_df, "protocol_a_median")
    create_OLS_model_for_VAD_and_other_features(all_committees_merged_df, "protocol_a_avg")
    # create_OLS_model_for_VAD_and_other_features(all_committees_merged_df, "protocol_v_var")
