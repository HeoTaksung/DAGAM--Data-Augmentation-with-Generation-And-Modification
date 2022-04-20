import re
import csv
import random
from transformers import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DAG_method():
    def __init__(self, input_file, ratio, tokenizer, model, output_file):
        self.input_file = input_file
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.class_sentence = None
        self.classes_name = None
        self.aug_sentence = None
        self.output_file = output_file
        
    def data_load(self):
        file = open(self.input_file, 'r', encoding='utf-8-sig')
        
        rdr = csv.reader(file)
        
        class_sentence = {}
        
        sentences = []
        labels = []
         # if 20newsgroup, TREC, R8, R52
        for line in rdr:
            line[1] = re.sub('[^a-zA-Z]', ' ', line[1]).strip().lower()
            line[1] = ' '.join(line[1].split())
            sentences.append(line[1])
            labels.append(line[0])

        file.close()

        X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.1, random_state=40, stratify=labels)

        for sentence, label in zip(X_train, y_train):
            if label not in class_sentence.keys():
                class_sentence[label] = []
            class_sentence[label].append(sentence)

        self.class_sentence = class_sentence
        self.classes_name = list(self.class_sentence.keys())
    
    
    def summary_extraction(self, texts):
        number_of_augmented_text = 0

        aug_sentence = []

        duplication_indexes = [] # indexes storage to avoid duplication of augmented text

        pbar = tqdm(total=int(len(texts) * self.ratio))

        while True:
            if number_of_augmented_text == int(len(texts) * self.ratio):
                break

            max_index = len(texts) - 1

            original_text_index1 = random.randint(0, max_index)
            original_text_index2 = random.randint(0, max_index)
            original_text_index3 = random.randint(0, max_index)

            if (original_text_index1 == original_text_index2) or (original_text_index1 == original_text_index3) or (original_text_index2 == original_text_index3): # To prevent the use of text having the same index
                continue

            if [original_text_index1, original_text_index2, original_text_index3] in duplication_indexes:
                continue

            duplication_indexes.append([original_text_index1, original_text_index2, original_text_index3])

            combining_text = texts[original_text_index1] + '\n' + texts[original_text_index2] + '\n' + texts[original_text_index3]

            inputs = self.tokenizer("summarize: " + combining_text, return_tensors="pt", truncation=True)
            outputs = self.model.generate(inputs["input_ids"], max_length=512, min_length=0, length_penalty=2.0, num_beams=4, early_stopping=True)
            aug_sentence.append(self.tokenizer.decode(outputs[0]).replace('<pad> ', "").replace('</s>', ""))

            pbar.update(1)
            number_of_augmented_text += 1

        pbar.close()

        return aug_sentence
    
    def augmentation(self):
        aug_sentence = {}

        for i in range(len(self.classes_name)):
            aug_sentence[self.classes_name[i]] = self.summary_extraction(self.class_sentence[self.classes_name[i]])

        self.aug_sentence = aug_sentence

    
    def savedFile(self):
        
        file = open(self.output_file, 'w', newline='')

        wdw = csv.writer(file)
        
        for class_name, sentence in zip(self.class_sentence.keys(), self.class_sentence.values()):
            for i in sentence:
                wdw.writerow([class_name, i])

        for class_name, sentence in zip(self.aug_sentence.keys(), self.aug_sentence.values()):
            for i in sentence:
                wdw.writerow([class_name, i])

        file.close()
        
    def run(self):
        self.data_load()
        self.augmentation()
        self.savedFile()