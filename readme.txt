this repo contains an dynamic hand gesture recognition model
this repo contains 4 maojor files
1. `data_collection.py` - contains the code to load the data and create the dataset
2. `data_preprocessing.py` - contains the code to preprocess the data. It reads all the files in datasets then perform change of origin on them and thenassign appropriate lables and save them as x , y and lable_map.
3. `model.py` - contains the lstm model architecture
4. `result.py` - contains the code to run and test the model

## How to run the code
1. First run the `data_collection.py` file to collect the data (run in venve)
2. Then run the `data_preprocessing.py` file to preprocess the data (run in venve)
3. Then run the `model.py` file to train the model (run in global env)
4. Then run the `result.py` file to test the model (run in venve)

along with tis there is also a `visualize.py` file which contains the code to visualize the data
