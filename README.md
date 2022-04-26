# Paying More Attention to Self-attention: Improving Pre-trained Language Models via Attention Guiding

by Shanshan Wang, Muyang Ma, Zhumin Chen, Zhaochun Ren, Huasheng Liang, Qiang Yan, Pengjie Ren
>>@article{wang2022paying,\
>>  title={Paying More Attention to Self-attention: Improving Pre-trained Language Models via Attention Guiding},\
>>  author={Wang, Shanshan and Ma, Muyang and Chen, Zhumin and Ren, Zhaochun and Liang, Huasheng and Yan, Qiang and Ren, Pengjie},\
>>  journal={arXiv preprint arXiv:2204.02922},\
>>  year={2022}\
>>}


## Running experiments

### Requirements
This code is written in PyTorch. Any version later than 1.9 is expected to work with the provided code. Please refer to the [official website](https://pytorch.org/) for an installation guide.

### Datasets
The datasets used in this work are [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) for natural language inference, [MedNLI](https://physionet.org/content/mednli/1.0.0/) for natural language inference on medical domain and [Cross-genre-IR](github.com/chzuo/emnlp2020-cross-genre-IR) for the across medical genres query. 
And then put the downloaded dataset in the `.data/` folder.

### Training
+ Run the following scripts for model training on different datasets:

```
python ./train/multiNLI/multiNLI_main_train.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
python ./train/medNLI/medNLI_main_train.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
python ./train/IR/IR_pair_main_train.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
```
### Evaluating
+ Run the following scripts for model evaluation on different datasets:
```
python ./test/multiNLI_test.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
python ./test/medNLI_test.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
python ./test/IR_pair_test.py --model_name='bert-base-uncased' --loss_type='task+both' --pd_factor=0.001 --ad_factor=0.001
```
