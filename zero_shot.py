from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import pipeline
import argparse
import tensorflow

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

labels = [
    "arts_&_culture",
    "business_&_entrepreneurs",
    "celebrity_&_pop_culture",
    "diaries_&_daily_life",
    "family",
    "fashion_&_style",
    "film_tv_&_video",
    "fitness_&_health",
    "food_&_dining",
    "gaming",
    "learning_&_educational",
    "music",
    "news_&_social_concern",
    "other_hobbies",
    "relationships",
    "science_&_technology",
    "sports",
    "travel_&_adventure",
    "youth_&_student_life"
]

#PARSER
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--facebook", help="use facebook/bart-large-mnli model. Default option", action="store_true")
parser.add_argument("-m2", "--model_2", help="use 2nd trained model", action="store_true")
parser.add_argument("-m3", "--model_3", help="use 3rd trained model", action="store_true")
parser.add_argument("-m4", "--model_4", help="use 4th trained model", action="store_true")
parser.add_argument("-s", "--sequence", help="sequence to be ran through model", type=str, default="The sky is blue, but soon the winter will come and the snow will cover the roofs.")
parser.add_argument("-fr", "--full_run", help="use selected model and perform a run through the whole dataset", action="store_true")
args = parser.parse_args()

if args.model_2:
    model = "/mnt/c/Users/vojta/Documents/BP/2023/BP-Topic-discovery-Neural-Network/models/model_2"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
elif args.model_3:
    model = "/mnt/c/Users/vojta/Documents/BP/2023/BP-Topic-discovery-Neural-Network/models/model_3"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
elif args.model_4:
    model = "/mnt/c/Users/vojta/Documents/BP/2023/BP-Topic-discovery-Neural-Network/models/model_4"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
else:
    model = "facebook/bart-large-mnli"  
    classifier = pipeline("zero-shot-classification", model=model)

#Only run one sample 
if args.full_run == False:
    
    results = classifier(args.sequence, labels, multi_label=True) 
    print(args.sequence)
    for i, score in enumerate(results["scores"]):
        print(results["labels"][i], ":", score)
    exit(0)

#Load dataset
dataset = load_dataset('cardiffnlp/tweet_topic_multi', split='test_2021')
texts = dataset["text"]

predictions = []
true_labels = []

#Go through the whole dataset
for i, text in enumerate(texts):
    results = classifier(text, labels, multi_label=True) 
    results["actual_topic"] = results["labels"].copy()
    
    #Add a list of actual topic values from dataset
    for index, value in enumerate(results["actual_topic"]):
        results["actual_topic"][index] = LABEL2ID[value]
        results["actual_topic"][index] = dataset["label"][i][results["actual_topic"][index]]
    print(text)
    for j, score in enumerate(results["scores"]):
        if score > 0.8:
            print(results["labels"][j], ":", score)
    
    #Append predictions and actual topics for loss calculation
    predictions.append(results["scores"])
    true_labels.append(dataset["label"][i])
    
#Calculate loss uing Binary cross entropy loss    
criterion = tensorflow.keras.losses.BinaryCrossentropy(reduction=tensorflow.keras.losses.Reduction.SUM) 
loss = criterion(true_labels, predictions)
average_loss = loss / tensorflow.cast(tensorflow.shape(predictions)[0], dtype=tensorflow.float32)

print("Average Binary Cross-Entropy Loss:", average_loss.numpy())