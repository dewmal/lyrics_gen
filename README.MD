## Lyrics Generator 
### PyTorch - Generative Adversarial Network  For Text Generate

#### Dataset
380,000+ lyrics from MetroLyrics
https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics

#### Requirements

* pandas - data import
* spacy - nlp
* pytorch - computation
* numpy - computation
* pytorch-ignite - manage training process

#### Getting Start

 1. ##### Download dataset

  * using kaggle api
  ```bash
    !kaggle datasets download -d gyani95/380000-lyrics-from-metrolyrics
    mkdir data
    mv 380000-lyrics-from-metrolyrics.zip data/
  ```
  2. ##### Python Requirements install and run
  ```bash
    pip install -r requirements.txt
    python app.py
  ```
