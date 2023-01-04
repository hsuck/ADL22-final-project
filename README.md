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
pip install -r requirements.txt
```

## Usage
```
usage: train_dssm.py [-h] [--output OUTPUT_FILE] [--model_path MODEL_PATH] [--user_file USER_FILE] [--course_file COURSE_FILE] [--vocab_path VOCAB_PATH] [--train_file TRAIN_FILE]
                     [--val_file VAL_FILE] [--test_file TEST_FILE] [--test] [--embed_size EMBED_SIZE] [--dnn DNN [DNN ...]] [--temp TEMP] [--dropout DROPOUT] [--lr LR]
                     [--weight_decay WEIGHT_DECAY] [--epoch EPOCH]

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
  --dnn DNN [DNN ...]   arch of user & course tower's DNN (default: [256, 128, 64])
  --temp TEMP           temperature (default: 1)
  --dropout DROPOUT     doupout rate (default: 0)
  --lr LR               learning rate (default: 0.0001)
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 1e-06)
  --epoch EPOCH         number of training epoch (default: 10)
```

## Training and Testing with default arguments
```
python3 train_dssm.py # training
python3 train_dssm.py --test # testing
```
