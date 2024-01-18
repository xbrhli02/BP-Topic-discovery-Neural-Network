from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import csv

LABEL2ID = {
    "arts_&_culture": 0,
    "business_&_entrepreneurs": 1,
    "celebrity_&_pop_culture": 2,
    "diaries_&_daily_life": 3,
    "family": 4,
    "fashion_&_style": 5,
    "film_tv_&_video": 6,
    "fitness_&_health": 7,
    "food_&_dining": 8,
    "gaming": 9,
    "learning_&_educational": 10,
    "music": 11,
    "news_&_social_concern": 12,
    "other_hobbies": 13,
    "relationships": 14,
    "science_&_technology": 15,
    "sports": 16,
    "travel_&_adventure": 17,
    "youth_&_student_life": 18
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def preprocess_function(text):
    tokenized_inputs = tokenizer(text, truncation=True, padding=True, max_length=128)
    return tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]
    
def predict(text):
    input_ids, attention_mask = preprocess_function(text)
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    logits = outputs.logits[0].tolist()
    predictions = [index for index, value in enumerate(logits) if value > -1.5]
    if not predictions:
        predictions = [logits.index(max(logits))]
    return predictions

model_path = "/mnt/c/Users/vojta/Documents/BP/2023/model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

dataset = load_dataset('cardiffnlp/tweet_topic_multi', split='test_2021')
texts = dataset["text"]

all_predictions = []
with open("predictions.csv", "w", newline='') as csvfile:
    fieldnames = ['Text', 'Prediction', 'Actual Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for i, text in enumerate(texts):
        predicted_class = predict(text)
        all_predictions.append(predicted_class)

        predicted_label = [ID2LABEL[index] for index in predicted_class]
        actual_label = dataset["label_name"][i]

        writer.writerow({'Text': text, 'Prediction': predicted_label, 'Actual Label': actual_label})

