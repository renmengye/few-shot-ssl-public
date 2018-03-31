# few-shot-ssl-public
Code for paper
*Meta-Learning for Semi-Supervised Few-Shot Classification.* [[arxiv](https://arxiv.org/abs/1803.00676)]

## Dependencies
* cv2
* numpy
* pandas
* python 2.7 / 3.5+
* tensorflow 1.3+
* tqdm

Our code is tested on Ubuntu 14.04 and 16.04.

## Setup
First, designate a folder to be your data root:
```
export DATA_ROOT={DATA_ROOT}
```

Then, set up the datasets following the instructions in the subsections.

### Omniglot
[[Google Drive](https://drive.google.com/open?id=1INlOTyPtnCJgm0hBVvtRLu5a0itk8bjs)]  (9.3 MB)
```
# Download and place "omniglot.tar.gz" in "$DATA_ROOT/omniglot".
mkdir -p $DATA_ROOT/omniglot
cd $DATA_ROOT/omniglot
mv ~/Downloads/omniglot.tar.gz .
tar -xzvf omniglot.tar.gz
rm -f omniglot.tar.gz
```

### miniImageNet
[[Google Drive](https://drive.google.com/open?id=16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY)]  (1.1 GB)
```
# Download and place "mini-imagenet.tar.gz" in "$DATA_ROOT/mini-imagenet".
mkdir -p $DATA_ROOT/mini-imagenet
cd $DATA_ROOT/mini-imagenet
mv ~/Downloads/mini-imagenet.tar.gz .
tar -xzvf mini-imagenet.tar.gz
rm -f mini-imagenet.tar.gz
```

### tieredImageNet
[[Google Drive](https://drive.google.com/open?id=1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u)]  (12.9 GB)
```
# Download and place "tiered-imagenet.tar" in "$DATA_ROOT/tiered-imagenet".
mkdir -p $DATA_ROOT/tiered-imagenet
cd $DATA_ROOT/tiered-imagenet
mv ~/Downloads/tiered-imagenet.tar .
tar -xvf tiered-imagenet.tar
rm -f tiered-imagenet.tar
```
Note: Please make sure that the following hardware requirements are met before running
tieredImageNet experiments.
* Disk: **30 GB**
* RAM: **32 GB**


## Core Experiments
Please run the following scripts to reproduce the core experiments.
```
# Clone the repository.
git clone https://github.com/renmengye/few-shot-ssl-public.git
cd few-shot-ssl-public

# To train a model.
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset {DATASET}                \
                  --label_ratio {LABEL_RATIO}        \
                  --model {MODEL}                    \
                  --results {SAVE_CKPT_FOLDER}       \
                  [--disable_distractor]

# To test a model.
python run_exp.py --data_root $DATA_ROOT             \
                  --dataset {DATASET}                \
                  --label_ratio {LABEL_RATIO}        \
                  --model {MODEL}                    \
                  --results {SAVE_CKPT_FOLDER}       \
                  --eval --pretrain {MODEL_ID}       \
                  [--num_unlabel {NUM_UNLABEL}]      \
                  [--num_test {NUM_TEST}]            \
                  [--disable_distractor]             \
                  [--use_test]
```
* Possible `{MODEL}` options are `basic`, `kmeans-refine`, `kmeans-refine-radius`, and `kmeans-refine-mask`.
* Possible `{DATASET}` options are `omniglot`, `mini-imagenet`, `tiered-imagenet`.
* Use `{LABEL_RATIO}` 0.1 for `omniglot` and `tiered-imagenet`, and 0.4 for `mini-imagenet`. 
* Replace `{MODEL_ID}` with the model ID obtained from the training program.
* Replace `{SAVE_CKPT_FOLDER}` with the folder where you save your checkpoints.
* Add additional flags `--num_unlabel 20 --num_test 20` for testing `mini-imagenet` and `tiered-imagenet` models, so that each episode contains 20 unlabeled images per class and 20 query images per class.
* Add an additional flag `--disable_distractor` to remove all distractor classes in the unlabeled images.
* Add an additional flag `--use_test` to evaluate on the test set instead of the validation set.
* More commandline details see `run_exp.py`.

## Simple Baselines for Few-Shot Classification
Please run the following script to reproduce a suite of baseline results.
```
python run_baseline_exp.py --data_root $DATA_ROOT    \
                           --dataset {DATASET}
```
* Possible `DATASET` options are `omniglot`, `mini-imagenet`, `tiered-imagenet`.

## Run over Multiple Random Splits
Please run the following script to reproduce results over 10 random label/unlabel splits, and test 
the model with different number of unlabeled items per episode. The default seeds are 0, 1001, ..., 
9009.
```
python run_multi_exp.py --data_root $DATA_ROOT       \
                        --dataset {DATASET}          \
                        --label_ratio {LABEL_RATIO}  \
                        --model {MODEL}              \
                        [--disable_distractor]       \
                        [--use_test]
```
* Possible `MODEL` options are `basic`, `kmeans-refine`, `kmeans-refine-radius`, and `kmeans-refine-mask`.
* Possible `DATASET` options are `omniglot`, `mini_imagenet`, `tiered_imagenet`.
* Use `{LABEL_RATIO}` 0.1 for `omniglot` and `tiered-imagenet`, and 0.4 for `mini-imagenet`. 
* Add an additional flag `--disable_distractor` to remove all distractor classes in the unlabeled images.
* Add an additional flag `--use_test` to evaluate on the test set instead of the validation set.

## Citation
If you use our code, please consider cite the following:
* Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle and Richard S. Zemel.
Meta-Learning for Semi-Supervised Few-Shot Classification. 
In *Proceedings of 6th International Conference on Learning Representations (ICLR)*, 2018.

```
@inproceedings{ren18fewshotssl,
  author   = {Mengye Ren and 
              Eleni Triantafillou and 
              Sachin Ravi and 
              Jake Snell and 
              Kevin Swersky and 
              Joshua B. Tenenbaum and 
              Hugo Larochelle and 
              Richard S. Zemel},
  title    = {Meta-Learning for Semi-Supervised Few-Shot Classification},
  booktitle= {Proceedings of 6th International Conference on Learning Representations {ICLR}},
  year     = {2018},
}
```
