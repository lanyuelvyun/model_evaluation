# -*- coding:utf-8 -*-
"""
@author: lisiqi
@time: 2019/7/19 11:30
@目的：比较新老模型，评估新模型相对于老模型的变化
"""
import pandas as pd
import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class CompareTwoModel(object):
    def __init__(self, df, bins_list_old, bins_list_new, save_path):
        """
        模型效果评估
        :param df: pd.DataFrame, at least contains old_model_score and new_model_score
        :param bins_list_old: the list of split value for old_model
        :param bins_list_new: the list of split value for new_model
        :param save_path:
        """
        self.df = df
        self.bins_list_old = bins_list_old
        self.bins_list_new = bins_list_new
        self.save_path = save_path
        self._get_bins()  # 分别对新老模型分进行分层，得到各自的分层区间，左闭右开

    def cal_diffscore(self):
        print("画新老模型分差值频数占比图".center(80, '*'))
        print("对于同一批用户，计算新老模型分数的差值:new_model_score - old_model_score")
        self.df["diff_score"] = self.df["new_model_score"] - self.df["old_model_score"]
        self.df["diff_score_bins"] = pd.cut(self.df["diff_score"], bins=10, right=False, include_lowest=True)
        df_result = self.df.groupby("diff_score_bins").apply(lambda x: x.shape[0] * 1.0 / self.df.shape[0])
        x_list = [str(i) for i in df_result.index]
        y_list = (list(df_result.values))
        plt.figure(figsize=(8, 5))
        plt.tick_params(labelsize=8)  # tick_params可设置坐标轴刻度值属性
        plt.bar(x_list, y_list, label='diff_score', width=0.5, bottom=0, facecolor='lightskyblue', alpha=0.5)
        plt.xticks(x_list, color='black', rotation=30)  # 横坐标旋转60度
        plt.title('diff_score of two model:new - old')
        plt.xlabel("diff_score")
        plt.ylabel("distribution")
        plt.legend()  # 用来显示图例
        plt.savefig(self.save_path + r"\model_diff_score.jpg")
        plt.close()

    def compare_old_new_model(self):
        print("计算偏移矩阵".center(80, '*'))
        print("对于同一批用户，比较新老模型的打分差异，构造偏移矩阵.")
        # 根据老模型分层进行分组，然后计算落在老模型A分层内的样本，有多少又同时落在新分层A/B/C。以此类推
        new_model_bins = self.df["new_model_bins"].unique().tolist()
        df_result = self.df.groupby("old_model_bins").apply(lambda x: self._cal_offset_matrix(x, new_model_bins)).reset_index(level=1, drop=True)
        df_result = df_result.sort_index(axis=1)  # 按照列名进行排序
        df_result.to_csv(self.save_path + r"\model_offset_matrix.csv")

    def _get_bins(self):
        print("先根据模型分，进行分箱".center(80, '*'))
        print("分别对新老模型分进行分层，得到各自的分层区间，左闭右开！")
        self.df["old_model_bins"] = pd.cut(self.df["old_model_score"],bins=self.bins_list_old,right=False,include_lowest=True)
        self.df["new_model_bins"] = pd.cut(self.df["new_model_score"],bins=self.bins_list_new,right=False,include_lowest=True)

    @staticmethod
    def _cal_offset_matrix(df_sub, new_model_bins):
        # 计算偏移矩阵
        marix_dict = {}
        for bin_level in new_model_bins:
            marix_dict[bin_level] = round(df_sub[df_sub["new_model_bins"] == bin_level].shape[0] * 1.0 / df_sub.shape[0], 4)
        return pd.DataFrame.from_dict(marix_dict, orient='index').T
        # 如果一个字典中，key的值只有一个（不是key的个数有1个），那么在使用pd.DataFrame.from_dict(marix_dict, orient='columns')value时就会报错，解决方式就是上面return后面的方式实现


if __name__ == '__main__':
    # 比较新老模型在同一批用户上的打分情况，构造偏移矩阵
    path = r"C:\Users\V-DZ-00255\Downloads"
    df = pd.read_csv(path + r"\zxdae_v11score_v2score_v2lakalascore_dt0708_2.csv")
    df = df.rename(columns={"aka_y_zx_score_v2": "old_model_score", "aka_y_zx_score": "new_model_score"})
    df = df[(df["old_model_score"].notnull()) & (df["new_model_score"].notnull())]
    # save_path = r"E:\work\2 mission\feature_engineer\result"
    save_path = path
    # 新老模型的分层区间
    # bins_list_old = [0, 450, 465, 486, 505, 535, 560, 999]  # get bins 注意：左闭右开
    bins_list_old = [0, 439, 459, 479, 499, 520, 999]  # get bins 注意：左闭右开
    bins_list_new = [0, 439, 459, 479, 499, 520, 999]
    compare_two_model = CompareTwoModel(df=df, bins_list_old=bins_list_old, bins_list_new=bins_list_new, save_path=save_path)
    compare_two_model.compare_old_new_model()
    compare_two_model.cal_diffscore()
