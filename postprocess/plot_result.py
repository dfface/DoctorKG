import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator


def loss_metric_plot_optimize(history, epochs, img_path, plot_title, y_label):
    train_loss = history['train_loss']
    eval_precision = history['eval_precision']
    eval_recall = history['eval_recall']
    eval_f1 = history['eval_f1']
    steps = history['step']

    show_every = len(steps) // epochs
    sparse_epochs = [None] * len(steps)
    print(show_every)
    print(len(sparse_epochs))
    sparse_epochs[show_every-1::show_every] = list(range(1, epochs+1))
    # 跳过 第1轮epoch 前面的 step 值
    steps = steps[show_every-1:]
    sparse_epochs = sparse_epochs[show_every-1:]
    train_loss = train_loss[show_every-1:]
    eval_precision = eval_precision[show_every-1:]
    eval_recall = eval_recall[show_every-1:]
    eval_f1 = eval_f1[show_every-1:]
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(steps, train_loss, label='Training')
    axes[0].set_xticks(steps)
    axes[0].set_xticklabels(sparse_epochs)
    axes[0].legend(loc='upper right')
    # axes[0].set_xlabel('Epochs', fontsize='large')
    axes[0].set_ylabel(y_label, fontsize='large')
    # axes[0].grid()
    axes[1].plot(steps, eval_precision, label='Precision')
    axes[1].plot(steps, eval_recall, label='Recall')
    axes[1].plot(steps, eval_f1, label='F1-score')
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(sparse_epochs)
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Num of Epochs', fontsize='large')
    axes[1].set_ylabel('Eval Precision, Recall, F1-score', fontsize='large')
    # axes[1].grid()
    axes[0].set_title(plot_title, fontsize='x-large')
    # 顺序很重要，图绘制完成之后，调整显示刻度，每隔 show_every 步显示一个格子
    xticks_major_locator = MultipleLocator(show_every)
    axes[0].xaxis.set_major_locator(xticks_major_locator)
    axes[1].xaxis.set_major_locator(xticks_major_locator)
    # 顺序仍然很重要，必须先保存再show，否则保存的是空白的
    plt.savefig(img_path)
    # plt.show()


def loss_metric_plot(history, epochs, img_path, plot_title, y_label):
    train_loss = history['train_loss']
    eval_precision = history['eval_precision']
    eval_recall = history['eval_recall']
    eval_f1 = history['eval_f1']
    steps = history['step']

    show_every = len(steps) // epochs
    sparse_epochs = [None] * len(steps)
    print(show_every)
    print(len(sparse_epochs))
    sparse_epochs[show_every-1::show_every] = list(range(1, epochs+1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(steps, train_loss, label='Training')
    axes[0].set_xticks(steps)
    axes[0].set_xticklabels(sparse_epochs)
    axes[0].legend(loc='upper right')
    # axes[0].set_xlabel('Epochs', fontsize='large')
    axes[0].set_ylabel(y_label, fontsize='large')
    # axes[0].grid()
    axes[1].plot(steps, eval_precision, label='Precision')
    axes[1].plot(steps, eval_recall, label='Recall')
    axes[1].plot(steps, eval_f1, label='F1-score')
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(sparse_epochs)
    axes[1].legend(loc='lower right')
    axes[1].set_xlabel('Num of Epochs', fontsize='large')
    axes[1].set_ylabel('Eval Precision, Recall, F1-score', fontsize='large')
    # axes[1].grid()
    axes[0].set_title(plot_title, fontsize='x-large')
    #plt.show()
    plt.savefig(img_path)


# 绘图时请保证 checkpoints/eval_results.csv 存在，并且训练是完全完成的，没有中断
eval_results_df = pd.read_csv("../checkpoints/eval_results.csv")
num_train_epochs = 30  # the number of training epochs
# loss_metric_plot(eval_results_df, num_train_epochs, "./result.pdf", "Bert for DoctorKG - Performance", "loss")
loss_metric_plot_optimize(eval_results_df, num_train_epochs, "./result.pdf", "BERT for DCCKG - Performance", "Loss")
