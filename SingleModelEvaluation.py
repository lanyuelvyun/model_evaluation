# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics
import datetime
import time


class BinsAnalysis(object):
    def __init__(self, df, split_col, bins_left, target_list, save_path):
        """
        对用户根据split_col的值进行分层，计算每一个分层内的逾期率和人数分布
        :param df: 原始数据，包含split_col
        :param split_col: 用于分箱的列
        :param bins_left: 每一个bin的左界值，是一个list
        :param target_list:
        :param save_path:
        """
        self.mode = mode
        self.df = df
        self.split_col = split_col
        self.bins_left = bins_left
        self.target_list = target_list
        self.save_path = save_path

        print("初始化开始".center(80, '*'))
        self._get_bins()  # 先分好层
        self._df_loan = self._get_loan_df()  # 根据账期维度得到订单维度
        print("初始化完成".center(80, '*'))
        self.get_bins_overdue_rate()

    def get_bins_overdue_rate(self):
        """计算各分层内的金额逾期率、cnt逾期率、cnt_rate"""
        # dt = datetime.datetime.strptime(self.dt, "%Y-%m-%d")
        for target in self.target_list:
            print("target = ", target)

            # 使用同一批用户，计算D0、D3、D7逾期率（取数的时候取出来有D7表现的用户，这部分用户D0、D3表现也都有）
            df_repay_sub = self.df
            print("df_repay_sub.shape ", df_repay_sub.shape)
            df_loan_sub = self._df_loan
            print("df_loan_sub.shape ", df_loan_sub.shape)

            print("calc AUC".center(20, '_'))
            auc = self._calc_auc(target, df_loan_sub)

            print("calc KS".center(20, '_'))
            ks, cut_list, tpr_list, fpr_list = self._calc_ks(target, df_loan_sub)

            print("计算每一层内的金额逾期率、笔数逾期率、笔数分布".center(20, '_'))
            index_list = df_repay_sub.groupby("bins").count().index
            columns_list = ["target", "loan_KS", "loan_AUC",
                            "repay_total_amt", "repay_due_amt", "repay_dueamt_rate",
                            "repay_total_cnt", "repay_due_cnt", "repay_duecnt_rate", "repay_cnt_distribution",
                            "loan_total_cnt", "loan_due_cnt", "loan_duecnt_rate", "loan_cnt_distribution"]
            df_result = pd.DataFrame(columns=columns_list, index=index_list)
            # 账期维度：每个分层内，金额逾期率
            result = df_repay_sub.groupby("bins").apply(lambda x: (x["due_amount"].sum(), x[x["overdue_day"] > target]["due_amount"].sum()))
            df_result["repay_total_amt"] = result.map(lambda x: x[0])
            df_result["repay_due_amt"] = result.map(lambda x: x[1])
            df_result["repay_dueamt_rate"] = df_result["repay_due_amt"] * 1.0 / df_result["repay_total_amt"]
            # 账期维度：笔数逾期率
            result = df_repay_sub.groupby("bins").apply(lambda x: (x.shape[0], x[x["overdue_day"] > target].shape[0]))
            df_result["repay_total_cnt"] = result.map(lambda x: x[0])
            df_result["repay_due_cnt"] = result.map(lambda x: x[1])
            df_result["repay_duecnt_rate"] = df_result["repay_due_cnt"] * 1.0 / (df_result["repay_total_cnt"] + 1e-20)
            # 账期维度：笔数分布
            total_cnt = df_repay_sub.shape[0]
            df_result["repay_cnt_distribution"] = df_result["repay_total_cnt"] * 1.0 / (total_cnt + 1e-20)
            # 订单维度：笔数逾期率
            result = df_loan_sub.groupby("bins").apply(lambda x: (x.shape[0], x[x["overdue_day"] > target].shape[0]))
            df_result["loan_total_cnt"] = result.map(lambda x: x[0])
            df_result["loan_due_cnt"] = result.map(lambda x: x[1])
            df_result["loan_duecnt_rate"] = df_result["loan_due_cnt"] * 1.0 / (df_result["loan_total_cnt"] + 1e-20)
            #  账期维度：笔数分布
            total_cnt = df_loan_sub.shape[0]
            df_result["loan_cnt_distribution"] = df_result["loan_total_cnt"] * 1.0 / (total_cnt + 1e-20)

            df_result["loan_AUC"] = auc
            df_result["loan_KS"] = ks
            df_result["target"] = target
            # save
            df_result[columns_list].sort_index(axis=1)
            df_result.to_csv(self.save_path, float_format="%.5f", mode='a')

    def _get_loan_df(self):
        print("get loan df")
        df_max = self.df.groupby("loan_id").max()[["user_id", "overdue_day"]]
        df_max.rename(columns={"overdue_day": "max_overdue"}, inplace=True)
        df_repay = pd.merge(self.df, df_max, left_on="loan_id", right_index=True, how="left")
        df_loan = df_repay.drop_duplicates(subset="loan_id", keep="first")  # 将loan_id去重
        print("df_loan.shape ", df_loan.shape)
        # df_loan.to_csv(r"E:\work\2 mission\3 Parsing_online_logs\calculate_auc_online\analysis_result\zxdae_acard\dt0512\zxdae_acardscore_v2_dt0512_loan.csv", index=None)
        return df_loan

    def _get_bins(self):
        print("get bins: 根据 %s 进行分箱，得到分箱区间，左闭右开！" % self.split_col)
        self.df["bins"] = pd.cut(self.df[self.split_col],
                                 bins=self.bins_left,
                                 right=False,  # 是否包含右区间
                                 include_lowest=True  # 是否包含左区间
                                )

    def _calc_auc(self, target, df_loan_sub):
        """
        1、注意计算AUC的时候，label与pvalue必须是正相关的！！！，否则计算的AUC就不对
        2、计算AUC，必须以loan_id为维度，因为在建模的时候就是以loan_id为维度
        """
        print("注意,计算AUC的时候，必须以loan_id为维度。并且label与pvalue必须是正相关的！否则计算的AUC就不对")
        df_loan_sub_copy = df_loan_sub.copy()
        # 去掉空值
        df_loan_sub_copy = df_loan_sub_copy[df_loan_sub_copy[self.split_col].notnull()]  # 模型分可能有NULL
        df_loan_sub_copy = df_loan_sub_copy[df_loan_sub_copy[self.split_col] >= 0]  # NULL值可能填充成-1，-2，-3
        df_loan_sub_copy["label"] = df_loan_sub_copy["max_overdue"].apply(lambda x: 1 if x <= target else 0)
        auc = metrics.roc_auc_score(df_loan_sub_copy["label"], df_loan_sub_copy[self.split_col]) 
        print("AUC = ", auc)
        return auc

    def _calc_ks(self, target, df_loan_sub):
        print("计算 %s 的ks" % self.split_col)
        print("注意,计算ks的时候，必须以loan_id为维度。并且该计算KS的逻辑适用于：p值越高，label=1,p_list范围是[300, 700]")
        # 去掉空值，NULL值可能填充成-1，-2，-3
        df_loan_sub_copy = df_loan_sub.copy()
        df_loan_sub_copy = df_loan_sub_copy[(df_loan_sub_copy[self.split_col].notnull()) & (df_loan_sub_copy[self.split_col] >= 0)]
        df_loan_sub_copy["label"] = df_loan_sub_copy["max_overdue"].apply(lambda x: 1 if x > target else 0)

        p_list = df_loan_sub_copy[self.split_col]
        label_list = df_loan_sub_copy["label"]
        tuple_list = list(zip(p_list, label_list))
        unique_value = sorted(np.unique(p_list))
        # cut_list = np.arange(0, 1, 0.002)
        cut_list = np.arange(300, 700, 1)
        if len(unique_value) < len(cut_list):
            cut_list = unique_value
        ks_thred = 0
        max_dist = 0
        init_value = 0.00001
        tpr_list = []
        fpr_list = []
        for cut in cut_list:
            tp = init_value
            fn = init_value
            fp = init_value
            tn = init_value
            for score, label in tuple_list:
                if score <= cut and label == 1:
                    tp += 1
                elif score <= cut and label == 0:
                    fp += 1
                elif score > cut and label == 1:
                    fn += 1
                elif score > cut and label == 0:
                    tn += 1
            tpr = round(1.0 * tp / (tp + fn), 3)
            fpr = round(1.0 * fp / (fp + tn), 3)
            dist = tpr - fpr
            if dist > max_dist:
                max_dist = dist  # max(tpr-fpr)，就是KS
                ks_thred = cut  # 取到max(tpr-fpr)时候的阈值p
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return round(max_dist, 5), cut_list, tpr_list, fpr_list


if __name__ == "__main__":
    path = r"C:\Users\V-DZ-00255\Downloads"

    # 统计数据：
    df = pd.read_csv(path + r"\quyong_zxdae_dt0716.csv", nan_values=[-1, -2, -3, -99])
    save_path = path + r"\quyong_zxdae_dt0716_result.csv"
    split_col = 'aka_qy_y_zx_score'　　# 模型分，用于分层
    print(df[split_col].min(), df[split_col].max())
    bins_left = [0, 455, 470, 490, 505, 520, 540, 999]  # 自定义的分层区间，按照这个区间进行分层。注意：左闭右开

    # 计算各分层的金额逾期率，单数逾期率，样本分布
    target_list = [0, 3, 7]　＃　分别计算Ｄ０、Ｄ３、Ｄ７逾期率
    binsanalysis = BinsAnalysis(df=df, split_col=split_col,
                                bins_left=bins_left, target_list=target_list, save_path=save_path)



