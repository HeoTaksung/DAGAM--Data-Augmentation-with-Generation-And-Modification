from transformers import *
from DAG_method import DAG_method
from DAM_method import DAM_method
from config import parser_return

args = parser_return()

method = args.method
input_file = args.input
output_file = args.output
ratio = args.ratio
num_sample = args.num_sample
model = args.model

if method == "DAG" or method == "DAGAM":
	tokenizer = AutoTokenizer.from_pretrained(model)
	generative_model = AutoModelForSeq2SeqLM.from_pretrained(model)

	DAG = DAG_method(input_file, ratio, tokenizer, generative_model, output_file)
	DAG.run()

	if method == "DAGAM":
		DAM = DAM_method(output_file, ratio, num_sample, output_file, DAGAM=True)
		DAM.run()

else:
	DAM = DAM_method(input_file, ratio, num_sample, output_file)
	DAM.run()
