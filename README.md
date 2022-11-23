In this project I am going to use LSTM-based neural networks to identify gender from names. I am using character embedding to pass to the LSTM model. Character embedding is suitable in this case because mostly differs based on gender by changing a few characters.
For dataset I am using the following link:

```
[a link] (https://archive.ics.uci.edu/ml/datasets/Gender+by+Name)
```

<img width="258" alt="Screen Shot 1401-09-02 at 21 08 43" src="https://user-images.githubusercontent.com/71961438/203613907-536ecb92-5a3a-49cf-9263-8079df931dd0.png">


## Steps 

> - Import required packages
> - Load data for the male and female names
> - Create a vocabulary
> - Create a data set class 
> - Create model 
> - Train the model
> - Analyze the results


## Read and Clean the Data
Dataset used is publicly available data from the following GitHub location. Its read in pandas dataframe with name and gender as columns. 

## Create Vocabulary of the characters 
I am planning to use character embedding, therefore, I need to convert every name into a set of character and each character is represented by a unique number.(CHAR2INDEX & INDEX2CHAR).
Then for each list i use padding(PAD) and UNK(for OOV),as well.Then we have two lists for lables(INDEX2LABEL & LABELTOINDEX) 


## Create a Dataset class
Pytorch dataset class is extended to return name and gender tensor pair for each data in the dataset.This class mapped the data for names and gender ,too.This is useful while training the model. 

## Create The Model
As i mentioned, I am using character embedding to pass to the LSTM model.
<img width="457" alt="Screen Shot 1401-09-02 at 21 08 46" src="https://user-images.githubusercontent.com/71961438/203613706-4a60b165-5b67-464f-a85a-b9d45543f0a3.png">


## Train the Model
We need a loss function as criteria and an optimizer to train our model. I am using SGD as my optimizer and BCELoss and loss function to train the model.

## Analyze The Results
For analyzing I am plotting the loss function for train and validation set. As we can see we don't have over/underfit.

