from transformers import *
from DAG_method import DAG_method
from DAM_method import DAM_method


input_file = './TREC-10/train.csv'
output_file = './train_aug_1.csv'

ratio = 5

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

DAG = DAG_method(input_file, ratio, tokenizer, model, output_file)
DAG.run()

DAM = DAM_method(input_file, ratio, 0.2, output_file, DAGAM=False)  # DAGAM=True > When used DAGAM
DAM.run()


# saved_training_data(saved_file_name, class_sentence, aug_sentence)