# DAGAM: Data Augmentation with Generation And Modification

## Text Augmentation

----------------------------------------

## Dataset

  * [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)

  * [AGnews](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

  * [20Newsgroups](http://qwone.com/~jason/20Newsgroups/)

  * [TREC](https://emilhvitfeldt.github.io/textdata/reference/dataset_trec.html)

  * [R8](http://www.daviddlewis.com/resources/testcollections/reuters21578/)

  * [R52](http://www.daviddlewis.com/resources/testcollections/reuters21578/)

----------------------------------------

## Method

  * DAG: used a generation model ([T5 base model](https://huggingface.co/t5-base)) by combining three texts instead of one, to provide variations to input data.

  * DAM: used Character order change (COC) strategy, which comes after the phenomenon, to text data. COC denotes fixing the first and the last character in a word and randomly permuting the rest.
