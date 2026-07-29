# NSQE

# README

**This repository accompanies our paper currently under review.**

## 1. Introduction
Natural Spectra Quality Evaluator (NSQE).<br>
A method for evaluating the authenticity of simulated Raman spectra. <br>
It assesses authenticity by calculating the Mahalanobis distance between the spectral dataset under evaluation and a reference multivariate Gaussian model of natural scene statistical features. <br>
This method aims to objectively evaluate the authenticity of simulated datasets by quantifying the statistical distance between the statistical distributions of the tested spectra and the true spectra.<br>
A smaller NSQE value indicates a closer resemblance to real spectra.<br>

## 2. Calculating NSQE
To compute the NSQE score, replace the paths in `NSQE.py` with the directories of the authentic baseline spectra and the spectral data to be evaluated:

    NSQE.py:
    baseline_npy_path = 'baseline_data.npy' # Data for calculating the benchmark MVG
    test_npy_path = 'test_dataset.npy'      # Data to be evaluated 

## 3. Directory Structure
    measured_dataset/: Authentic microplastic spectral datasets.
    simulated_dataset/: Simulated spectral datasets.
    run_reconstruction/: Directory for saving models during the training process of the spectral reconstruction task.
    run_classification/: Directory for saving models during the training process of the classification task.
Data Availability: The authentic and simulated datasets, alongside the experimental results supporting this study, are available via ScienceDB: https://www.scidb.cn/s/nI3IFb

## 4. Training and Prediction Scripts
    train_reconstruction.py: Training script for the spectral reconstruction (denoising and baseline correction) models.
    fit_reconstruction.py: Script for invoking trained reconstruction models to process additional simulated spectral data.
    predict_reconstruction.py: Apply trained model on measured data, plot 8-subplot comparison, and save results.
    train_classification.py: Automated iterative training script for the microplastic spectral classification models.
    predict_classification.py: Script for invoking trained models to process authentic spectral data, as well as testing and cross-evaluating the classification models.
    SE_ResUNet.py: Defines all neural network architectures utilized in this project, encompassing the ResUNet for spectral reconstruction and the SE-ResNet for classification.
    tools.py: Utility library.

## 5.Generative Model Evaluation Metrics Calculation
    fit_kld.py: Script for calculating the one dimensional Kullback-Leibler Divergence (KLD) evaluation metric.
    fit_kid.py: Script for calculating the one dimensional Kernel Inception Distance (KID) evaluation metric.
    fit_fid.py: Script for calculating the one dimensional Fréchet Inception Distance (FID) evaluation metric.
    fit_is.py: Script for calculating the one dimensional Inception Score (IS) evaluation metric.
    run/: Directory containing the pretrained classifier models utilized for feature extraction.
    
## 6.Other
    tSNE_FigS3.py: Plot for FigS3.
## 7.Output Result Tables
    Batch_Evaluation_Results.xlsx: Performance of models trained on simulated datasets 1 to 12 when evaluated against the additional test set (dataset_extra_reconstruction).
    NSQE_Correlation coefficient.xlsx: Correlations between the evaluation scores of simulated datasets 1 to 12 and the regression metrics of the additional test set (dataset_extra_reconstruction).
    generative model assessment metrics.xlsx: Evaluation scores for simulated datasets 13 to 24.

# README

**这个存储库伴随着我们正在审稿的论文**

## 1.简介
Natural Spectra Quality Evaluator NSQE。 一种用于评估模拟拉曼光谱真实度的自然光谱质量评价方法。<br>
计算待评估光谱数据集与真实光谱的自然场景统计特征的基准多元高斯模型的马氏（Mahalanobis）距离评估真实度。<br>
该方法旨在通过量化待测光谱与真实光谱统计分布之间的统计距离，来客观评价模拟数据集的真实度。<br>
更小的 NSQE 值代表模拟光谱的自然统计特征越接近真实光谱，真实度越高。<br>

## 2.计算NSQE
在NSQE.py将下面的路径换成作为基准的真实光谱数据和待评估的光谱的数据：<br>

    NSQE.py:
    baseline_npy_path = 'baseline_data.npy' #计算基准MVG的数据
    test_npy_path = 'test_dataset.npy'      #待计算的数据

## 3.文件夹
    measured_dataset/:实测微塑料光谱数据集
    simulated_dataset/:模拟光谱数据集
    run_reconstruction/:用于保存光谱重建任务训练过程中的模型
    run_classification/: 用于保存分类任务训练过程中的模型
这项论文的实测数据集和模拟数据集，以及实验运行结果可以通过scienceDB获取: https://www.scidb.cn/s/nI3IFb
## 4.训练与预测脚本
    train_reconstruction.py: 光谱重建（去噪与基线校正）模型的训练脚本
    fit_reconstruction.py: 调用训练好的重建模型，对额外模拟光谱数据进行处理
    predict_reconstruction.py: 应用训练好的模型对实测数据进行去噪，绘制 8 子图对比并导出预测结果
    train_classification.py: 微塑料光谱分类模型的自动循环训练脚本
    predict_classification.py: 调用训练好的重建模型，对实测光谱数据进行处理，分类模型的测试与交叉评估脚本
    SE_ResUNet.py: 定义了项目中使用的所有神经网络架构，包含用于光谱重建的 ResUNet 以及用于分类的 SE-ResNet 
    tools.py: 工具库。

## 5.生成模型评价指标计算
    fit_kld.py:用于计算一维 Kullback-Leibler Divergence (KLD) 评价指标
    fit_kid.py:用于计算一维 Kernel Inception Distance (KID) 评价指标
    fit_fid.py:用于计算一维 Fréchet Inception Distance (FID) 评价指标
    fit_is.py:用于计算一维 Inception Score (IS) 评价指标
    run/:提取特征的预训练分类器模型
    
## 6.其他
    tSNE_FigS3.py:用于FigS3 t-SNE绘图
    
## 7.输出结果表格
    Batch_Evaluation_Results.xlsx: 模拟数据集1-12训练的模型在额外测试集(dataset_extra_reconstruction)的性能
    NSQE_Correlation coefficient.xlsx: 模拟数据集1-12的评估分数与额外测试集(dataset_extra_reconstruction)回归指标的相关性
    generative model assessment metrics.xlsx: 模拟数据集13-24的评估分数
