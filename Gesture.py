from data_collection import collect_data
from data_preprodessing import preprocess_data
from model import lstm_model
from result import load_model

choice = input("Do you want to collect more data? (y/n): ")
while choice.lower() == "y":
    collect_data()
    preprocess_data()

choice = input("Do you want to Train model? (y/n): ")
if choice.lower() == "y":
    lstm_model()

load_model()
