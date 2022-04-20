import re
import csv
import random
from transformers import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def data_load(file_name):
    file = open(file_name, 'r', encoding='utf-8-sig')

    rdr = csv.reader(file)

    class_sentence = {}

    sentences = []
    labels = []
    # if 20newsgroup, TREC, R8, R52
    for line in rdr:
        sentences.append(re.sub('[^a-zA-Z]', ' ', line[1]).strip().lower())
        labels.append(line[0])
        
    file.close()

    X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.1, random_state=40, stratify=labels)

    for sentence, label in zip(X_train, y_train):
        if label not in class_sentence.keys():
            class_sentence[label] = []
        class_sentence[label].append(sentence)

    return class_sentence


def summary_extraction(n, texts, tokenizer, model):
    number_of_augmented_text = 0
    
    aug_sentence = []
    
    duplication_indexes = [] # indexes storage to avoid duplication of augmented text
    
    pbar = tqdm(total=int(len(texts) * n))
    
    while True:
        if number_of_augmented_text == int(len(texts) * n):
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
        
        inputs = tokenizer("summarize: " + combining_text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=512, min_length=0, length_penalty=2.0, num_beams=4, early_stopping=True)
        aug_sentence.append(tokenizer.decode(outputs[0]).replace('<pad> ', "").replace('</s>', ""))
        
        pbar.update(1)
        number_of_augmented_text += 1
        
    pbar.close()
    
    return aug_sentence


def DAG(n, class_sentence, tokenizer, model):
    classes_name = list(class_sentence.keys())
    aug_sentence = {}

    for i in range(len(classes_name)):
        aug_sentence[classes_name[i]] = summary_extraction(n, class_sentence[classes_name[i]], tokenizer, model)
    
    return aug_sentence


def string_shuffle(sentence):
    character_change = []
    
    for i in range(1, len(sentence)-1):
        character_change.append(i)
        
    random.shuffle(character_change)
    
    return character_change


def DAM(n, num_sample, class_sentence):
    aug_sentence = {}
    
    for labels in class_sentence.keys():
        for sentence, label in tqdm(zip(class_sentence[labels], labels), total = len(list(class_sentence.values()))):
            sent = sentence.split()
            number_of_augmented_text = 0
            end_while = 0 # if charater length of all words != 4  then break
            while True:
                if number_of_augmented_text == n or end_while == 20:
                    break
                coc_sentence = ""
                index = -1
                for sen in sent:
                    random_sample = random.sample(
                        [x for x in range(len(sent))], int(len(sent) * num_sample))
                    index += 1
                    if len(sen) >= 4 and index in random_sample:
                        coc_sentence += sen[0]
                        shuffle_list = string_shuffle(sen)
                        for i in range(len(shuffle_list)):
                            coc_sentence += sen[shuffle_list[i]]
                        coc_sentence += sen[-1]
                    else:
                        coc_sentence += sen
                    coc_sentence += " "
                coc_sentence = coc_sentence.strip()
                end_while += 1
                
                if label not in aug_sentence.keys():
                    aug_sentence[label] = []
                if coc_sentence in aug_sentence[label]:
                    continue

                aug_sentence[label].append(coc_sentence)
                number_of_augmented_text += 1

    return aug_sentence

    
def saved_training_data(saved_file_name, class_sentence, aug_sentence):
    classes_name = list(class_sentence.keys())
    file = open(saved_file_name, 'w', newline='')

    wdw = csv.writer(file)

    for class_name, sentence in zip(class_sentence.keys(), class_sentence.values()):
        for i in sentence:
            wdw.writerow([class_name, i])

    for class_name, sentence in zip(aug_sentence.keys(), aug_sentence.values()):
        for i in sentence:
            wdw.writerow([class_name, i])

    file.close()