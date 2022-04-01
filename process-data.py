#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：cat-fight-dog
@File ：process-data.py
@Author ：wanghao
@Date ：2022/4/1 21:04
@Description:  将训练集随机划分10%成验证数据集
"""
import os
import glob
import random
import shutil


# 如果没有路径则写一个路径
def mkdirs_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_data(rate=0.1):
    src = './dataset/train'

    # 1.检查并创建文件夹
    val_src = './dataset/val'
    mkdirs_if_missing(val_src)

    # 2.抽样 获取文件夹下所有jpg文件
    img_pths = glob.glob(src + os.sep + '**.jpg')
    random.shuffle(img_pths)  # 打乱顺序
    val_count = int(len(img_pths) * rate)
    val_pths = random.sample(img_pths, val_count)  # 抽样
    # 3.移动抽样的图片到验证集
    for pth in val_pths:
        name = pth.split(os.sep)[-1]
        shutil.move(pth, os.path.join(val_src, name))
    print(f'验证集样本数量为：{val_count}')


if __name__ == '__main__':
    split_data(0.1)
