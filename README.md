## introduction

This project replicates the classic paper [Attention is All You Need](https://arxiv.org/abs/1706.03762), selecting the English-to-Chinese translation task and using the [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) dataset. Since the primary goal of this project is to reproduce the paper and validate certain ideas, many engineering steps such as checkpointing and monitoring have been omitted. It’s important to note that the code does not save the model.

Due to limited resources, training was conducted for only 9 hours on a single 4090 GPU. Since the results met expectations, the full training process was not completed.

## quick glance

```text
python ./train.py
Using device: cuda
Device name: NVIDIA GeForce RTX 4090
Device memory: 23.64971923828125 GB
Using the latest cached version of the dataset since Helsinki-NLP/opus-100 couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'en-zh' at data/Helsinki-NLP___opus-100/en-zh/0.0.0/805090dc28bf78897da9641cdf08b61287580df9 (last modified on Sat Mar  8 09:56:48 2025).
Using the latest cached version of the dataset since Helsinki-NLP/opus-100 couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'en-zh' at data/Helsinki-NLP___opus-100/en-zh/0.0.0/805090dc28bf78897da9641cdf08b61287580df9 (last modified on Sat Mar  8 09:56:48 2025).
Using the latest cached version of the dataset since Helsinki-NLP/opus-100 couldn't be found on the Hugging Face Hub
Found the latest cached dataset configuration 'en-zh' at data/Helsinki-NLP___opus-100/en-zh/0.0.0/805090dc28bf78897da9641cdf08b61287580df9 (last modified on Sat Mar  8 09:56:48 2025).
Loading tokenizer for en
Loading tokenizer for zh
90.222896 M parameters
Processing eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:18<00:00, 25.55it/s]
step 20000: val loss 0.3533████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:18<00:00, 24.40it/s]
Processing eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:22<00:00, 24.21it/s]
step 40000: val loss 0.2920████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:22<00:00, 24.25it/s]
Processing eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:19<00:00, 25.28it/s]
step 60000: val loss 0.2840████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:19<00:00, 26.00it/s]
 69%|██████████████████████████████████████████████████████████████████████▋                                | 68666/100000 [6:19:27<2:46:26,  3.14it/s, loss=0.160]Processing eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:19<00:00, 25.11it/s]
step 80000: val loss 0.2890████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:19<00:00, 24.65it/s]
Processing eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:18<00:00, 25.62it/s]
step 99999: val loss 0.2985████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [01:18<00:00, 24.89it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [9:09:24<00:00,  3.03it/s, loss=0.175]
---------- 1 ----------
source_text: He commended UNCTAD's concerted efforts in assisting the Palestinian Authority and expressed his support for UNCTAD's proposed new activities, especially in the area of food security, trade facilitation, and transport and supply.
target_text: 他赞扬贸发会议在援助巴勒斯坦权力机构方面的协调一致的努力，并表示支持贸发会议拟议的新活动，特别是在粮食安全、简化贸易手续和运输与供应方面的活动。
model_out_text: 29 . 他 赞扬 贸发会议 在 援助 巴勒斯坦权力机构 方面所 做的 协同 努力 ， 支持 贸发会议 提议 的新 活动 ， 即 食品 安全 、 国际贸易 便利化 、 运输和 供应 领域 。

---------- 2 ----------
source_text: I'll take one of the girls, a married one with kids.
target_text: 我要选女人 结婚有小孩的 警察要是杀了她们
model_out_text: 我先 带 她 两个 小孩 ， 也就是 个 孩子 ， 一个 已婚 的 女人

---------- 3 ----------
source_text: That was quite a ride.
target_text: 一路那个闹腾啊
model_out_text: 这 蛮 喜欢 的一个

---------- 4 ----------
source_text: J. Strengthening African statistical systems to generate gender-disaggregated data to support policies to promote gender equality and empowerment of women
target_text: J. 加强非洲的统计系统，以编制按性别分列的数据，支持各项政策，促进两性平等及 提高妇女能力
model_out_text: J . 加强 非洲 统计 系统 ， 编写 基于性别 分列 的数据 支持 促进 两性平等和 赋予妇女权力 的政策

---------- 5 ----------
source_text: Achieving their expeditious entry into force and global implementation is a major task for IMO in order to advance the fight against international terrorism, and their implementation is referenced in more than one planned output in the Organization's High-Level Action Plan.
target_text: 议定书尽快生效并得到普遍执行，是海事组织在推动打击国际恐怖主义方面的重大任务，海事组织《高级行动计划》几个计划产出都提到了这两项议定书的执行工作。
model_out_text: 3 . 为了实现 一旦 进入 全球 的 形势 下 ， 海事组织 的一项 重大 任务 就是 ， 推动 打击国际恐怖主义 的斗争 ， 其 执行 受到 本组织 迄今为止 一个 高级别 行动计划 中的一 项目标 产出 具体 规定的 归 类 。
```

## quick start

```bash
git clone https://github.com/GHOST-LOVE-YOU/pytorch-transformer.github
cd pytorch-transformer
# optional start | linux only
python -m venv myenv
source myenv/bin/activate
# optional end
pip install -r requirements.txt
python ./train.py
```

## thanks

- [Umar Jamil](https://www.youtube.com/watch?v=ISNdQcPhsts)
- [Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
