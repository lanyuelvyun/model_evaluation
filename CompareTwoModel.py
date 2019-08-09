# -*- coding:utf-8 -*-
"""
@author: lanyue
@time: 2019/08/09 16:25
@目的：比较新老模型分的效果
"""
import pandas as pd
import random
import re
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class CompareTwoModel(object):
    def __init__(self, df, score_info_dict, save_path):
        """
        模型效果评估
        :param df: pd.DataFrame, at least contains old_model_score、new_model_score and label
        :param score_info_dict: {"old_score":[200, 300, 400, 500], "new_score1":[200, 300, 400, 500]} 模型分名字：分层区间
        :param save_path:
        """
        print("蓝月善意提醒".center(80, '-'))
        print("df at least contains old_model_score、new_model_score and label!!!!")
        # 去掉空值,模型分可能有NULL
        self.df = df
        self.score_info_dict = score_info_dict
        self.save_path = save_path
        self._get_bins()  # 分别根据新老模型分对样本进行分层，得到各个模型的分层区间，左闭右开

    def plot_score_distribution(self):
        df_good = self.df[self.df["label"] == 0]
        df_bad = self.df[self.df["label"] == 1]
        print("df_good.shape: ", df_good.shape, "df_bad.shape:", df_bad.shape)
        plt.figure(figsize=(20, 12))
        color_list = ["blue", "red", "yellow", "green", "orange", "black"]
        j = 0
        for score_name in self.score_info_dict.keys():
            plt.subplot(241 + j)
            sns.kdeplot(df_good[score_name], label='%s,label=0' % score_name, color=color_list[j], linestyle='-', shade=True)
            sns.kdeplot(df_bad[score_name], label='%s,label=1' % score_name, color=color_list[j], linestyle='--', shade=True)
            # plt.xticks(x_list, color='black', rotation=60)  # 横坐标旋转60度
            plt.title('score_distribution')
            plt.xlabel("score")
            plt.ylabel("frequency")
            plt.legend()        # 用来显示图例
            plt.tight_layout()  # 调整每个子图之间的距离
            j += 1
        plt.savefig(self.save_path + r"\score_distribution.jpg")
        plt.close()

    def cal_diffscore(self):
        print("同一批样本的情况下：画新老模型分差值(new-old)频数占比图".center(80, '-'))
        df_copy = self.df.copy()
        plt.figure(figsize=(20, 12))
        j = 0
        for score_name in self.score_info_dict.keys():
            if score_name not in (["old_score"]):
                df_copy["diff_score"] = df_copy[score_name] - df_copy["old_score"]
                df_copy["diff_score_bins"] = pd.cut(df_copy["diff_score"], bins=10, right=False, include_lowest=True)
                df_result = df_copy.groupby("diff_score_bins").apply(lambda x: x.shape[0] * 1.0 / self.df.shape[0])
                # plot
                plt.subplot(241 + j)
                j += 1
                x_list = [str(i) for i in df_result.index]
                y_list = (list(df_result.values))
                plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
                plt.bar(x_list, y_list, label='diff_score', width=0.5, bottom=0, facecolor='blue', alpha=0.5)
                plt.xticks(x_list, color='black', rotation=30)  # 横坐标旋转60度
                plt.title('diff_score = %s - old_score' % score_name)
                plt.xlabel("diff_score")
                plt.ylabel("distribution")
                plt.legend()  # 用来显示图例
                plt.tight_layout()  # 调整每个子图之间的距离
        plt.savefig(self.save_path + r"\model_diff_score.jpg")
        plt.close()

    def cal_offset_matrix(self):
        print("计算偏移矩阵".center(80, '-'))
        print("对于同一批用户，比较新老模型的打分差异，构造偏移矩阵,以老模型为基准.")
        # 根据老模型分层进行分组，然后计算落在老模型A分层内的样本，有多少又同时落在新分层A/B/C。以此类推
        for score_name, split_value_list in self.score_info_dict.items():
            if score_name not in (["old_score"]):
                bins_name = score_name + '_bins'
                new_score_bins = self.df[bins_name].unique().tolist()
                df_result = self.df.groupby("old_score_bins").apply(lambda x: self._cal_offset_matrix(x, bins_name, new_score_bins)).reset_index(level=1, drop=True)
                df_result = df_result.sort_index(axis=1)  # 按照列名进行排序
                df_result.index.name = 'old_score_bins' + "/" + bins_name  # 对index的名字进行重命名
                df_result.to_csv(self.save_path + r"\model_offset_matrix.csv", mode='a')  # 追加保存

    def _get_bins(self):
        print("先根据模型分，进行分箱".center(80, '-'))
        print("分别根据新老模型分对样本进行分箱，得到各自的分箱区间，左闭右开！")
        for score_name, split_value_list in self.score_info_dict.items():
            bins_name = score_name + '_bins'
            self.df[bins_name] = pd.cut(self.df[score_name], bins=split_value_list, right=False, include_lowest=True)

    @staticmethod
    def _cal_offset_matrix(df_sub, bins_name, new_model_bins):
        # 计算偏移矩阵
        marix_dict = {}
        for bin in new_model_bins:
            marix_dict[bin] = round(df_sub[df_sub[bins_name] == bin].shape[0] * 1.0 / (df_sub.shape[0] + 1e-20), 4)
        return pd.DataFrame.from_dict(marix_dict, orient='index').T
        # 如果一个字典中，每一个key的值只有一个，那么在使用pd.DataFrame.from_dict(marix_dict, orient='columns')时就会报错，解决方式就是orient='index'，然后再进行转置。

    def cal_auc(self):
        """
        计算AUC，必须以建模时候的最小粒度样本来进行计算。
        入参：模型分+label
        1、建模的时候，label的定义：坏用户label=1, 好用户label=0
        2、注意计算AUC的时候，函数metrics.roc_auc_score(label_list, pvalue_list)的label与pvalue必须是这样的对应关系：pvalue越高，用户越坏(label=1)，否则计算的AUC就不对
        3、但是我们现在有的模型分：取值范围在[300， 700]之间，分数越高，用户越好(label=0)，那怎么计算AUC呢？
        4、解决办法：将label颠倒过来，好用户label=1, 坏用户label=0。将这个label_list和模型分送入函数metrics.roc_auc_score()，就可以了。
        """
        print("calc AUC".center(60, '-'))
        print("注意,必须以建模时候的最小粒度样本来进行计算。并且label与pvalue必须是正相关的！否则计算的AUC就不对")
        df_copy = self.df.copy()
        df_copy["label_convert"] = df_copy["label"].apply(lambda x: 1 if x == 0 else 0)  # 把label反过来
        plt.figure(figsize=(10, 8))
        for score_name in self.score_info_dict.keys():
            auc = metrics.roc_auc_score(df_copy["label_convert"], df_copy[score_name])
            fpr, tpr, threshold = metrics.roc_curve(df_copy["label_convert"], df_copy[score_name])
            plt.plot(fpr, tpr, label='roc_curve_%s (area = %0.3f)' % (score_name, auc))
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="upper left")  # 用来显示图例
        plt.savefig(os.path.join(self.save_path, 'roc_curve.png'))
        plt.close()

    def cal_ks(self):
        print("calc ks and plot ks_cusrve".center(60, '-'))
        df_copy = self.df.copy()
        plt.figure(figsize=(10, 8))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        color_list = ["blue", "red", "yellow", "green", "orange", "black"]
        for i, score_name in enumerate(self.score_info_dict.keys()):
            ks, ks_thred, cut_list, tpr_list, fpr_list = self._calc_ks(df_copy[score_name], df_copy["label"])
            # plot
            plt.plot(cut_list, tpr_list, label='tpr_%s,ks = %s' % (score_name, ks), color=color_list[i], linestyle='-')
            plt.plot(cut_list, fpr_list, label='fpr_%s' % score_name, color=color_list[i], linestyle='-')
            plt.plot([ks_thred, ks_thred], [0, 1], label='ks_thred_%s' % score_name, color=color_list[i], linestyle='--')
        plt.plot([400, 600], [0, 1], color='black', linestyle='--')
        # plt.xticks(x_list, color='black', rotation=60)  # 横坐标旋转60度
        plt.title('ks_curve')
        plt.xlabel("score")
        plt.ylabel("tpr/fpr")
        plt.legend()  # 用来显示图例
        path = os.path.join(self.save_path, 'ks_curve.png')
        plt.savefig(path)
        plt.close()
        # return ks_old, ks_new

    @staticmethod
    def _calc_ks(p_list, label_list):
        """
        计算ks的时候，必须以建模时候的最小粒度样本来进行计算
        需要：模型分+label
        1、以下计算KS的逻辑适用于：模型分数范围是[300, 700]，分数越高，用户越好(label=0)
        2、label的定义：与模型训练时候一致，坏用户label=1, 好用户label=0
        """
        print("计算ks，以下计算KS的逻辑适用于：模型分数范围是[300, 700]，分数越高，用户越好(label=0)")
        tuple_list = list(zip(p_list, label_list))
        unique_value = sorted(np.unique(p_list))
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
        return round(max_dist, 5), ks_thred, cut_list, tpr_list, fpr_list

if __name__ == '__main__':
    # 比较新老模型在同一批用户上的打分情况，构造偏移矩阵
    # V1
    # path = r"C:\Users\V-DZ-00255\Downloads"
    # save_path = path
    # df = pd.read_csv(path + r"\v1_score_filled&old.csv")
    # df = df.drop_duplicates(subset="loan_id", keep="first")  # 去重loan_id
    # df['label'] = df["max_overdue"].apply(lambda x: 1 if x > 7 else 0)
    # print("df.shape", df.shape)
    # # 新老模型分，以及各自对应的分层区间。注意老模型分必须叫做“old_score”!!
    # df = df.rename(columns={"aka_y_zx_score_v1": "old_score"})
    # score_info_dict = {"old_score": [0, 465, 486, 505, 535, 560, 999],
    #                    "v1_score_new_20%": [0, 465, 486, 505, 535, 560, 999],
    #                    "v1_score_new_40%": [0, 465, 486, 505, 535, 560, 999],
    #                    "v1_score_new_50%": [0, 465, 486, 505, 535, 560, 999],
    #                    "v1_score_new_60%": [0, 465, 486, 505, 535, 560, 999]}

    # V2
    path = r"C:\Users\V-DZ-00255\Downloads"
    save_path = path
    df = pd.read_csv(path + r"\v2_score_filled&old.csv")
    df = df.drop_duplicates(subset="loan_id", keep="first")  # 去重loan_id
    df['label'] = df["max_overdue"].apply(lambda x: 1 if x > 7 else 0)
    print(df.shape)
    # 新老模型分，以及各自对应的分层区间。注意老模型分必须叫做“old_score”!!
    df = df.rename(columns={"aka_y_zx_score_v2": "old_score"})
    score_info_dict = {"old_score": [0, 439, 459, 479, 499, 520, 999],
                       "v2_score_new_20%": [0, 439, 459, 479, 499, 520, 999],
                       "v2_score_new_40%": [0, 439, 459, 479, 499, 520, 999],
                       "v2_score_new_50%": [0, 439, 459, 479, 499, 520, 999],
                       "v2_score_new_60%": [0, 439, 459, 479, 499, 520, 999]}

    # compare model
    compare_two_model = CompareTwoModel(df=df, score_info_dict=score_info_dict, save_path=save_path)
    compare_two_model.cal_offset_matrix()
    compare_two_model.cal_ks()
    compare_two_model.cal_auc()
    compare_two_model.cal_diffscore()
    compare_two_model.plot_score_distribution()
