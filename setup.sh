#!/usr/bin/env bash
echo "Shall get you up and running in no time"

echo "Lets first clone mytorch. We'll be using it rather unsparingly."
git clone https://github.com/geraltofrivia/mytorch.git -q
cd mytorch
chmod +x setup.sh
./setup.sh
cd ..


echo "Let's clone fastai, too (and use its old stuff)"
git clone https://github.com/fastai/fastai.git -q
pip install -r ./fastai/old/requirements.txt
mkdir tmp
mv fastai/old/fastai/* tmp/
rm -rf fastai/*
mv tmp/* fastai/
rm -rf tmp

# Download IMDB
echo "Okay time to download the raw data. \n We begin with IMDB"
mkdir raw
mkdir raw/imdb
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -P ./raw/imdb/
tar -zxvf ./raw/imdb/aclImdb_v1.tar.gz -C ./raw/imdb/

# Download Wikitext103
mkdir raw/wikitext
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P ./raw/wikitext/
unzip ./raw/wikitext/wikitext-103-v1.zip -d ./raw/wikitext/

# Download pretrained model
echo "Finally, lets download the pretrained model and we're good to go."
mkdir resources
wget -nH -r -np -P ./resources/ http://files.fast.ai/models/wt103/


mkdir resources/proc
mkdir resources/proc/imdb
mkdir resources/proc/imdb/data
mkdir resources/proc/imdb/datalm
mkdir resources/proc/imdb/model
mkdir resources/proc/imdb/dann_trim_default
mkdir resources/proc/imdb/dann_full_default
mkdir resources/proc/imdb/phase2_trim_default
mkdir resources/proc/imdb/phase2_full_default
