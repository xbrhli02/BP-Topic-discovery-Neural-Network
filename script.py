from deep_translator import GoogleTranslator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy 
from scipy.special import expit

MODEL_NAME = f"cardiffnlp/tweet-topic-21-multi"
#MODEL_NAME = f"cardiffnlp/tweet-topic-19-multi"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
class_mapping = model.config.id2label

""" two datasets to choose from: either sensitive topics or tweets
    in case of russian sensitive topics, also translate the text for better understanding."""

for text in load_dataset('NiGuLa/Russian_Sensitive_Topics', split='train')['text']:
#for text in load_dataset('cardiffnlp/tweet_topic_multi', split='train_2021')['text']:

    text = GoogleTranslator(source='ru', target='en').translate(text)    
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    scores = output[0][0].detach().numpy()
    scores = expit(scores)
    
    count = 0
    
    #Filter out texts with only one topic
    topic_count = [x for x in scores if x > 0.5]
    if(len(topic_count) < 2):
        continue
    
    #print otu
    print(text)
    for i in range(len(scores)):
        if scores[i] >= 0.5:
            count += 1
            print(count, f"({scores[i]:.2f})", class_mapping[i])
    print("\n")
    
