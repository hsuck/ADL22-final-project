# ADL22-final-project

## Installation
```
/* create conda environment */
conda create --name <env_name> python=3.9
conda activate <env_name>

/* install torch-rechub */
git clone https://github.com/datawhalechina/torch-rechub.git
patch -p0 < ./dssm.patch
cd torch-rechub
python setup.py install

/* install other packages */
cd ..
pip install -r requirements.txt
```

## Preprocessing
Before training, please go to [python/utils/](https://github.com/hsuck/ADL22-final-project/tree/ricky/python/utils) to perform preprocessing.
```
cd python/ultils
python3 vocab.py
```
Note: default path to put vocabs is `cache/vocab`
## Usage
All scripts are in `python/`
```
usage: train_dssm.py [-h] [--output OUTPUT_FILE] [--model_path MODEL_PATH] [--user_file USER_FILE] [--course_file COURSE_FILE] [--vocab_path VOCAB_PATH] [--train_file TRAIN_FILE]
                     [--val_file VAL_FILE] [--test_file TEST_FILE] [--test] [--embed_size EMBED_SIZE] [--dnn DNN [DNN ...]] [--temp TEMP] [--dropout DROPOUT] [--lr LR]
                     [--weight_decay WEIGHT_DECAY] [--epoch EPOCH] [--batch_size BATCH_SIZE]

Generates personalized recommendations for each user

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT_FILE  output file name (default: predictions.csv)
  --model_path MODEL_PATH
                        model saved directory (default: models/)
  --user_file USER_FILE
                        user data path (default: ../data/users.csv)
  --course_file COURSE_FILE
                        course data path (default: ../data/courses.csv)
  --vocab_path VOCAB_PATH
                        the path of vocab directory (default: ../cache/vocab)
  --train_file TRAIN_FILE
                        train data path (default: ../data/train.csv)
  --val_file VAL_FILE   val data path (default: ../data/val_seen.csv)
  --test_file TEST_FILE
                        test data path (default: ../data/test_seen.csv)
  --test
  --embed_size EMBED_SIZE
                        user & course's embedding size (default: 16)
  --dnn DNN [DNN ...]   user & course tower's DNN (default: [256, 128, 64])
  --temp TEMP           temperature (default: 1)
  --dropout DROPOUT     doupout rate (default: 0)
  --lr LR               learning rate (default: 0.0001)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-06)
  --epoch EPOCH         number of training epoch (default: 10)
  --batch_size BATCH_SIZE
```
Note: When using option `--dnn`, you need to pass a sequence, e.g. --dnn 128 64
```
usage: merge.py [-h] [--output OUTPUT_FILE] [--input INPUT_FILE] [--data_path DATA_PATH]

Append the number of subgroups to 50 for each user

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT_FILE  output file name (default: subgroups.csv)
  --input INPUT_FILE    the path of course prediction file (default: ./predictions.csv)
  --data_path DATA_PATH
                        the path of data directory (default: ../data/)
```

## Training and Testing with default arguments
```
python3 train_dssm.py # training
python3 train_dssm.py --test # testing

python3 merge.py # change course_id to corresponding subgroup for each user
```

## Reproduce
### Seen domain 
#### Course Prediction
```
python3 train_dssm.py --embed_size 16 --temp 1 --dropout 0.2 --epoch 10 --weight_decay 1e-5 # training
python3 train_dssm.py --embed_size 16 --dropout 0.2 --test --output prediction.csv # testing
```
#### Topic Prediction
```
python3 mergy.py --input prediction.csv --data_path {data path} --output subgroup.csv # need to predict course first
```

### Uneen domain 
#### Course Prediction
```
python3 train_dssm.py --embed_size 256 --temp 1 --epoch 10 --weight_decay 1e-5 # training
python3 train_dssm.py --embed_size 256 --test --output prediction.csv # testing
```
#### Topic Prediction
```
python3 mergy.py --input prediction.csv --data_path {data path} --output subgroup.csv # need to predict course first
```
