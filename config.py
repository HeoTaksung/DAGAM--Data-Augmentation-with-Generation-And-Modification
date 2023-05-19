import argparse

def parser_return():
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", type=str, default="DAM", help="DAM, DAG, DAGAM")
	parser.add_argument("--input", type=str, default=None, help="location of input data file")
	parser.add_argument("--output", type=str, default=None, help="location of output data file")
	parser.add_argument("--ratio", type=int, default=3, help="ratio of data to augment")
	parser.add_argument("--num_sample", type=float, default=0.2, help="percentage of words to apply Character Order Change")
	parser.add_argument("--model", type=str, default=None, help="generative model")

	return parser.parse_args()
	