# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics

class BinsAnalysis(object):
    def __init__(self, df, split_col, split_value_list, target_list, save_path):
        """
        模型评估：对用户根据模型分进行分层，计算每一层内的金额逾期率、笔数逾期率、笔数占比
        :param df: 原始统计数据，以账期为最小粒度，包含账期id，模型分、该账期编号（第几期）、放款金额、应还日期、逾期天数
        :param split_col: 用于分箱的列，就是模型分，范围在[300， 700]之间，分数越高，信用越好
        :param split_value_list: 分箱的分界值，是一个list
        :param target_list: 要统计的逾期目标，D0、D3或者D7逾期
        :param save_path: 保存结果的路径
        """
        self.df = df
        self.split_col = split_col
        self.split_value_list = split_value_list
        self.target_list = target_list
        self.save_path = save_path

        print("初始化开始".center(80, '*'))
        self._get_bins()  # 先分好层
        self._df_loan = self._get_loan_df()  # 根据账期维度得到订单维度
        print("初始化完成".center(80, '*'))
        self.get_bins_overdue_rate()

    def get_bins_overdue_rate(self):
        """使用同一批用户，计算各分层内的D0、D3、D7金额逾期率、笔数逾期率、人数占比（取数的时候取出来有D7表现的用户，这部分用户D0、D3表现也都有）"""
        for target in self.target_list:
            print("target = ", target)

            df_repay_sub = self.df
            print("df_repay_sub.shape ", df_repay_sub.shape)
            df_loan_sub = self._df_loan
            print("df_loan_sub.shape ", df_loan_sub.shape)

            print("calc AUC".center(20, '_'))
            auc = self._calc_auc(target, df_loan_sub)

            print("calc KS".center(20, '_'))
            ks, cut_list, tpr_list, fpr_list = self._calc_ks(target, df_loan_sub)

            print("计算每一层内的金额逾期率、笔数逾期率、笔数占比".center(20, '_'))
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
        print("得到以订单为最小粒度的df, 该订单的逾期天数=max(该订单全部有表现账期的逾期天数)")
        df_max = self.df.groupby("loan_id").max()[["user_id", "overdue_day"]]
        df_max.rename(columns={"overdue_day": "max_overdue"}, inplace=True)
        df_repay = pd.merge(self.df, df_max, left_on="loan_id", right_index=True, how="left")
        df_loan = df_repay.drop_duplicates(subset="loan_id", keep="first")  # 将loan_id去重
        print("df_loan.shape ", df_loan.shape)
        # df_loan.to_csv(r"result\loan.csv", index=None)
        return df_loan

    def _get_bins(self):
        print("get bins: 根据 %s 进行分箱，得到分箱区间，左闭右开！" % self.split_col)
        self.df["bins"] = pd.cut(self.df[self.split_col],
                                 bins=self.split_value_list,
                                 right=False,  # 是否包含右区间
                                 include_lowest=True  # 是否包含左区间
                                )

    def _calc_auc(self, target, df_loan_sub):
        """
        计算AUC，必须以loan_id为维度，因为在建模的时候就是以loan_id为维度
        需要：模型分+label
        1、建模的时候，label的定义：坏用户label=1, 好用户label=0
        2、注意计算AUC的时候，函数metrics.roc_auc_score(label_list, pvalue_list)的label与pvalue必须是这样的对应关系：pvalue越高，用户越坏(label=1)，否则计算的AUC就不对
        3、但是我们现在有的模型分：取值范围在[300， 700]之间，分数越高，用户越好(label=0)，那怎么计算AUC呢？
        4、解决办法：将label颠倒过来，好用户label=1, 坏用户label=0。将这个label_list和模型分输入到函数metrics.roc_auc_score()里面，就可以了。
        """
        print("注意,计算AUC的时候，必须以loan_id为维度。并且label与pvalue必须是正相关的！否则计算的AUC就不对")
        df_loan_sub_copy = df_loan_sub.copy()
        # 去掉空值
        df_loan_sub_copy = df_loan_sub_copy[df_loan_sub_copy[self.split_col].notnull()]  # 模型分可能有NULL
        df_loan_sub_copy = df_loan_sub_copy[df_loan_sub_copy[self.split_col] >= 0]  # NULL值可能填充成-1，-2，-3
        # label：由于模型分数越高，用户越好，所以label这样定义
        df_loan_sub_copy["label"] = df_loan_sub_copy["max_overdue"].apply(lambda x: 1 if x <= target else 0)
        auc = metrics.roc_auc_score(df_loan_sub_copy["label"], df_loan_sub_copy[self.split_col])
        print("AUC = ", auc)
        return auc

    def _calc_ks(self, target, df_loan_sub):
        """
        计算ks的时候，必须以loan_id为维度，因为在建模的时候就是以loan_id为维度
        需要：模型分+label
        1、以下计算KS的逻辑适用于：模型分数范围是[300, 700]，分数越高，用户越好(label=0)
        2、label的定义：与模型训练时候一致，坏用户label=1, 好用户label=0
        """
        print("计算ks，以下计算KS的逻辑适用于：模型分数范围是[300, 700]，分数越高，用户越好(label=0)")
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
    path = r"C:\Users\Downloads"
    df = pd.read_csv(path + r"\quyong_zxdae_dt0716.csv", na_values=[-1, -2, -3, -99])
    save_path = path + r"\quyong_zxdae_dt0716_result.csv"
    split_col = 'aka_qy_y_zx_score'  # 模型分，用于分层
    print(df[split_col].min(), df[split_col].max())
    split_value_list = [0, 455, 470, 490, 505, 520, 540, 999]  # 自定义的分层区间，按照这个区间进行分层。注意：左闭右开

    # 计算各分层的金额逾期率，单数逾期率，样本分布
    target_list = [0, 3, 7]  # 分别计算Ｄ０、Ｄ３、Ｄ７逾期率
    binsanalysis = BinsAnalysis(df=df, split_col=split_col,
                                split_value_list=split_value_list, target_list=target_list, save_path=save_path)
