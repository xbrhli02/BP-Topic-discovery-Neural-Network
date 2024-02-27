from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate
import numpy as np
import argparse
import torch

#Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#Compute predictions
def compute_metrics(eval_pred):
    print(eval_pred)
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    predictions = np.argmax(logits, axis=-1).tolist()
    labels = labels.tolist()
    return {"predictions": predictions, "references": labels}


#PARSER
parser = argparse.ArgumentParser()
parser.add_argument("-od", "--output_dir", help="directory for training output", type=str, default="OUTPUT_DIR")
parser.add_argument("-ld", "--logging_dir", help="directory for logs from training output", type=str, default="LOGGING_DIR")
parser.add_argument("-lr", "--learning_rate", help="learning rate for model training", type=float, default=2e-5)
parser.add_argument("-e", "--num_of_epochs", help="number of epochs for model training", type=int, default=5)
parser.add_argument("-wd", "--weight_decay", help="weight decay for model training", type=float, default=0.05)
parser.add_argument("-m", "--model_name", help="desired name of the model", type=float, default="TEST_MODEL")

args = parser.parse_args()

#load dataset and tokenizer
tweets = load_dataset('cardiffnlp/tweet_topic_multi')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

tokenized_tweets = tweets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_tweets["train_2021"]
small_eval_dataset = tokenized_tweets["test_2021"]

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=19, 
                                                           problem_type="multi_label_classification", 
                                                           id2label=ID2LABEL, label2id=LABEL2ID)
#Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    logging_dir=args.logging_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=args.num_of_epochs,
    weight_decay=args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("/mnt/c/Users/vojta/Documents/BP/2023/model_4")