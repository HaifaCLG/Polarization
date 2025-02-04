import pandas as pd
import numpy as np
from scipy.stats import kendalltau, spearmanr
import itertools
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from aux_functions import load_object
from vad_functions import select_knesset_sentences


def calculate_kendalls_w(data):
    # Number of objects and raters
    n = len(data)  # Number of objects (rows)
    m = len(data.columns)  # Number of raters (columns)

    # Calculate the sum of squared deviations from the mean ranks (S)
    mean_ranks = data.mean(axis=1)
    squared_deviations = ((data.T - mean_ranks) ** 2).sum().sum()
    S = squared_deviations
    # Calculate Kendall's W
    W = (12 * S) / (m ** 2 * (n ** 3 - n))

    return W


def calculate_pairwise_tau(data):
    all_pairs_taus = []
    for comb in itertools.combinations(columns, 2):
        print(comb)
        combination = list(comb)
        rankings = data[combination]
        # kendalls_w = calculate_kendalls_w(rankings)
        # print(f'kendalls_w results for {VAD_value}')
        # print(kendalls_w)
        tau, p_value = kendalltau(data[comb[0]], data[comb[1]])
        all_pairs_taus.append(tau)
        print(f"Kendall's tau 'tau': {tau}, 'p_value': {p_value}")
        print()
    mean_pairwise_tau = np.mean(all_pairs_taus)
    print(f'{VAD_value} mean pairwise tau: {mean_pairwise_tau}')


if __name__ == '__main__':
    VAD_value = "A"
    file_path = f'{VAD_value}-shira-israel-avia-ranks.csv'
    data = pd.read_csv(file_path)

    # Extract the average rankings adjusting for ties
    # columns = ['shira_rank_avg_tie', 'israel_rank_avg_tie', 'avia_rank_avg_tie']
    columns = ['shira_rank_avg_tie', 'israel_rank_avg_tie', 'avia_rank_avg_tie']

    rankings = data[columns]
    print(columns)

    kendalls_w = calculate_kendalls_w(rankings)
    print(f'kendalls_w results for {VAD_value}')
    print(kendalls_w)
    print()
    calculate_pairwise_tau(data)

    print(f"calculating kendall's tau between dicta-bert and avg-raters-score" )
    tau, p_value = kendalltau(data["annotators-avg-rank"], data["dictabert-binom-rank"])
    print(f'tau: {tau}, p-value: {p_value}')

    print(f"calculating kendall's tau between our-bert and avg-raters-score")
    tau, p_value = kendalltau(data["our-bert-rank"],data["annotators-avg-rank"])
    print(f'tau: {tau}, p-value: {p_value}')

    print(f"calculating Mean Squared Error (MSE) between our-bert and avg-raters-score")
    o_mse = mean_squared_error(np.array(data["avg score"]), np.array(data["our-bert-score"]))
    print("Mean Squared Error (MSE):", o_mse)

    print(f"calculating Mean Squared Error (MSE) between dicta-bert and avg-raters-score")
    d_mse = mean_squared_error(np.array(data["avg score"]), np.array(data["dictabert-binom score"]))
    print("Mean Squared Error (MSE):", d_mse)

    print(f"calculating Root Mean Squared Error (RMSE) between our-bert and avg-raters-score")
    o_rmse  = np.sqrt(o_mse)
    print("Root Mean Squared Error (RMSE):", o_rmse)

    print(f"calculating Root Mean Squared Error (RMSE) between dicta-bert and avg-raters-score")
    d_rmse = np.sqrt(d_mse)
    print("Root Mean Squared Error (RMSE):", d_rmse)

    print(f"calculating R-squared Score between our-bert and avg-raters-score")
    o_r2  = r2_score(np.array(data["avg score"]), np.array(data["our-bert-score"]))
    print("R-squared Score:", o_r2 )

    print(f"calculating R-squared Score between dicta-bert and avg-raters-score")
    d_r2 = r2_score(np.array(data["avg score"]), np.array(data["dictabert-binom score"]))
    print("R-squared Score:", d_r2)


    # print(f"calculating kendall's tau between our-bert only surface forms and avg-raters-rank")#TODO restore if want only surface forms
    # tau, p_value = kendalltau(data["our-bert-rank-only-surface-forms"], data["annotators-avg-rank"])
    # print(f'tau: {tau}, p-value: {p_value}')
    #
    # print(f"calculating kendall's tau between dicta-bert only surface forms and avg-raters-rank")
    # tau, p_value = kendalltau(data["dictabert-only-surface-forms-binom-rank"], data["annotators-avg-rank"])
    # print(f'tau: {tau}, p-value: {p_value}')

    if VAD_value == "V":
        print(f"calculating kendall's tau between our-bert-checkpoint-7427 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-7427-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-4244 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-4244-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-12732 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-12732-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-3183 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-3183-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-5305 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-5305-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-6366 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-6366-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-8488 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-8488-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-9549 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-9549-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-10610 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-10610-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between our-bert-checkpoint-11671 full and avg-raters-rank")
        tau, p_value = kendalltau(data["our-checkpoint-11671-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between multilingual full and avg-raters-rank")
        tau, p_value = kendalltau(data["multilingual-rank"], data["annotators-avg-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

        print(f"calculating kendall's tau between multilingual full and gili-rank")
        tau, p_value = kendalltau(data["multilingual-rank"], data["gili-rank"])
        print(f'tau: {tau}, p-value: {p_value}')

    # print(f"calculating kendall's tau between our-bert and dicta-bert-rank")
    # tau, p_value = kendalltau(data["dictabert-binom-rank"], data["our-bert-rank"])
    # print(f'tau: {tau}, p-value: {p_value}')

    res = pearsonr(data["our-bert-score"], data["avg score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(f"pearson's correlation  on avg annotators scores and our-bert-score: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

    res = pearsonr(data["dictabert-binom score"], data["avg score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation  on avg annotators scores and dicta-bert-score: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")


    if VAD_value == "V":
        res = pearsonr(data["our-checkpoint-7427-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-7427-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")


        res = pearsonr(data["our-checkpoint-4244-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-4244-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-12732-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-12732-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-3183-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-3183-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-5305-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-5305-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-6366-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-6366-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-8488-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-8488-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-9549-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-9549-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-10610-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-10610-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["our-checkpoint-11671-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and our-checkpoint-11671-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        res = pearsonr(data["multilingual-score"], data["avg score"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on avg annotators scores and multilingual-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

        print(f'calcultaion spearman between multi and annotators scores:')
        print(f'spearmanr: {spearmanr(data["avg score"], data["multilingual-score"])}')
        print(f'calcultaion spearman between dicta and annotators scores:')
        print(f'spearmanr: {spearmanr(data["avg score"],data["dictabert-binom score"])}')

        res = pearsonr(data["multilingual-score"], data["gili-scores"])
        statistic = res.statistic
        pvalue = res.pvalue
        print(
            f"pearson's correlation  on gili scores and multilingual-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")


        print(f'calcultaion spearman between multi and gili scores:')
        print(f'spearmanr: {spearmanr(data["gili-scores"],data["multilingual-score"])}')

    print(f"calculating kendall's tau between multilingual full and avg-raters-rank")
    tau, p_value = kendalltau(data["multilingual-rank"], data["annotators-avg-rank"])
    print(f'tau: {tau}, p-value: {p_value}')

    res = pearsonr(data["multilingual-score"], data["avg score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation  on avg annotators scores and multilingual-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

    print(f'calcultaion spearman between multi and annotators scores:')
    print(f'spearmanr: {spearmanr(data["avg score"], data["multilingual-score"])}')

    res = pearsonr(data["multilingual-score"], data["normalized_avg_score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation  on Normalized avg annotators scores and multilingual-score full: statistic: {'%.2f' % statistic}, pvalue: {'%.2f' % pvalue}")

    all_knesset_sentences, all_knesset_sentences_vad_values_df = select_knesset_sentences(1)
    trained_knesset_sentences = load_object(f'knesset_training_sentences')
    test_sentences = [sent for sent in all_knesset_sentences if sent not in trained_knesset_sentences]
    only_test_data = data[data['sent_text'].isin(test_sentences)]
    ordered_test_data_df = only_test_data.set_index('sent_text').loc[test_sentences].reset_index()


    res = pearsonr(ordered_test_data_df["multilingual-score"], ordered_test_data_df["avg score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation on test avg annotators scores and multilingual-score full: statistic: {'%.3f' % statistic}, pvalue: {'%.3f' % pvalue}")

    res = pearsonr(ordered_test_data_df["multilingual-score"], ordered_test_data_df["normalized_avg_score"])
    statistic = res.statistic
    pvalue = res.pvalue
    print(
        f"pearson's correlation on test Normalized avg annotators scores and multilingual-score full: statistic: {'%.3f' % statistic}, pvalue: {'%.3f' % pvalue}")

    # print(f'calcultaion spearman between multi and annotators scores:')
    # print(f'spearmanr: {spearmanr(ordered_test_data_df["avg score"], ordered_test_data_df["multilingual-score"])}')


