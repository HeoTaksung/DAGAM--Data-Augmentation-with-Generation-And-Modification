from posixpath import sep
import pandas as pd
import os
import random
import re
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv

class DAM_method():
    def __init__(self, file_name, ratio, num_sample, output_file, DAGAM=False):
        self.file_name = file_name
        self.ratio = ratio
        self.num_sample = num_sample
        self.output_file = output_file
        self.DAGAM = DAGAM

    def load_file(self):
        file_name = self.file_name
        
        file = open(file_name, 'r', encoding='utf-8-sig')
        
        sentence_list = []
        label_list = []
        
        rdr = csv.reader(file)
        
        for line in rdr:
            sentence_list.append(line[1])
            label_list.append(line[0])
        file.close()
        
        if self.DAGAM == False:
            train_sentence, val_sentence, train_label, val_label = train_test_split(sentence_list, label_list, test_size=0.1, random_state= 40)
            return train_sentence, train_label

        return sentence_list, label_list

    def cleanText(self, readData): # preprocessing text
        text = re.sub('[^a-zA-Z]', ' ', readData).strip().lower()
        text = ' '.join(text.split())
        return text

    def string_shuffle(self, string):
        word_shuffle = []
        for i in range(1, len(string)-1):
            word_shuffle.append(i)
        random.shuffle(word_shuffle)
        return word_shuffle

    
    def create_data_augmentation(self):
        sentences, labels = self.load_file()
        
        sentence_list, labels_list = [], []
        for sentence, label in tqdm(zip(sentences, labels), total = len(sentences)):
            sentence = sentence.replace("\t", " ")
            sentence = self.cleanText(sentence)
            sentence_list.append(sentence)
            labels_list.append(label)

            sent = sentence.split()
            for i in range(self.ratio):
                aug = ""
                index = -1
                for sen in sent:
                    random_sample = random.sample(
                        [x for x in range(len(sent))], int(len(sent) * self.num_sample))  # Choose random words to apply the Character Order Change Method
                    index += 1
                    if len(sen) >= 4 and index in random_sample: # Applies only when there are 4 or more words in a sentence
                        # shuffling except first word and last word
                        aug += sen[0]
                        shuffle_list = self.string_shuffle(sen)  #
                        for i in range(len(shuffle_list)):
                            aug += sen[shuffle_list[i]]
                        aug += sen[-1]
                    else:
                        aug += sen
                    aug += " "
                aug = aug.strip()
                if aug in sentence_list :
                    continue
                sentence_list.append(aug)
                labels_list.append(label)
        return sentence_list, labels_list
    
    
    
    def saveFile(self):
        sentence, label = self.create_data_augmentation()
        file = open(self.output_file, 'w', newline='')

        wdw = csv.writer(file)
        
        for i in range(len(sentence)):
            wdw.writerow([label[i], sentence[i]])
            
        file.close()

    def run(self):
        self.saveFile()