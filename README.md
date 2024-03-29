# Text Augmentation 

  * [DAGAM: Data Augmentation with Generation And Modification](https://arxiv.org/abs/2204.02633)
  
    * `Byeong-Cheol Jo`, `Tak-Sung Heo`, `Yeongjoon Park`, `Yongmin Yoo`, `Won Ik Cho`, `Kyungsun Kim`

---------------------------------------------

## Dataset

  * [IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)

  * [AGnews](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)

  * [20Newsgroups](http://qwone.com/~jason/20Newsgroups/)

  * [TREC](https://emilhvitfeldt.github.io/textdata/reference/dataset_trec.html)

  * [R8](http://www.daviddlewis.com/resources/testcollections/reuters21578/)

  * [R52](http://www.daviddlewis.com/resources/testcollections/reuters21578/)

---------------------------------------------

## Method

  * Data Augmentation with Generation (DAG)
    
    * Used a generation model ([T5 base model](https://huggingface.co/t5-base)) by combining three texts instead of one, to provide variations to input data.

  * Data Augmentation with Modification (DAM)

    * Used Character order change (COC) strategy, which comes after the phenomenon, to text data. COC denotes fixing the first and the last character in a word and randomly permuting the rest.

  * Data Augmentation with Generation And Modification (DAGAM)

    * Used combining two strategies of DAG and DAM.
    
 -------------------------------------------
 
 ## Using Augmentation
  * `$ pip3 install -r requirements.txt`
 
  * Data Augmentation with Generation
  
    * `$ python3 main.py --method="DAG" --input="input_file_path" --output="output_file_path" --ratio=3 --num_sample=0.2  --model="t5-base"`
      * ratio: ratio of data to augment
      * num_sample: percentage of words to apply Character Order Change Method
      * model: generative model
      
      
  * Data Augmentation with Modification
  
    * `$ python3 main.py --method="DAM" --input="input_file_path" --output="output_file_path" --ratio=3 --num_sample=0.2`
      * ratio: ratio of data to augment
      * num_sample: percentage of words to apply Character Order Change Method
  

   
  * Data Augmentation with Generation And Modification
  
    * `$ python3 main.py --method="DAGAM" --input="input_file_path" --output="output_file_path" --ratio=3 --num_sample=0.2 --model="t5-base"`
      * ratio: ratio of data to augment
      * num_sample: percentage of words to apply Character Order Change Method
      * model: generative model
