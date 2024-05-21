import numpy as np
from scipy.stats import spearmanr, kendalltau, rankdata

# Calculate Kendall's and Spearman's coefficients
def get_corr_coeff(pred_imp_scores, videos, dataset, user_scores=None):
    rho_coeff, tau_coeff = [], []
    if dataset=='SumMe':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            true = np.mean(user_scores,axis=0)
            rho_coeff.append(spearmanr(pred_imp_score,true)[0])
            tau_coeff.append(kendalltau(rankdata(pred_imp_score),rankdata(true))[0])
    elif dataset=='TVSum':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            pred_imp_score = np.squeeze(pred_imp_score).tolist()
            user = int(video.split("_")[-1])

            curr_user_score = user_scores[user-1]

            tmp_rho_coeff, tmp_tau_coeff = [], []
            for annotation in range(len(curr_user_score)):
                true_user_score = curr_user_score[annotation]
                curr_rho_coeff, _ = spearmanr(pred_imp_score, true_user_score)
                curr_tau_coeff, _ = kendalltau(rankdata(pred_imp_score), rankdata(true_user_score))
                tmp_rho_coeff.append(curr_rho_coeff)
                tmp_tau_coeff.append(curr_tau_coeff)
            rho_coeff.append(np.mean(tmp_rho_coeff))
            tau_coeff.append(np.mean(tmp_tau_coeff))
    rho_coeff = np.array(rho_coeff).mean()
    tau_coeff = np.array(tau_coeff).mean()
    return rho_coeff, tau_coeff