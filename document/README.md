# Question Classifier

Git repo: [Question Classifier](https://github.com/buchuitoudegou/QuestionClassifier)

## pipeline
```
tokenization->word embedding->sentence vector->training the classifier
```

## commit msg
`[your task]: what you did in this commit`

e.g.: 'wordEmbedding: word2vec model initialize'

...

## environment
`Developing and testing environment`: macOS10.15.7, Anaconda python3.8, with 8-gen Core i5 CPU and 16GB RAM.

Training set: [5500-labeled questions](https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label)

Testing set: [TREC 10](https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label)

## run
```
mkdir data/models
cd src
```
Preprocess data using `--preprocess` flag. Please make sure preprocessing has been done before running training.
```
python3 question_classifier.py --preprocess --config [config-file-path]
```
dev training mode:
Leaving 10% of training set out as validation set
```
python3 question_classifier.py --dev --config [config-file-path]
```
training mode:
Train the model with the whole dataset
```
python3 question_classifier.py --train --config [config-file-path]
```
test mode:
Read an existing model and test it on TREC 10 dataset
```
python3 question_classifier.py --test --config [config-file-path]
```
