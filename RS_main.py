# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:09:21 2018

@author: Samaneh
"""

import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from collections import defaultdict

from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

#############################################################################################################
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
4
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
#############################################################################################################
def precision_recall_at_k(predictions, k=10, threshold=2):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

#############################################################################################################

# Prepare the data with unix timestamp format and add rate 1 for each intraction
mydata = pd.read_csv('.\data_rating_timestamp.csv')

# Make rate column for each user-item
mydata_pivot1 = pd.pivot_table(mydata, index=['user','item'], values=['rating'], aggfunc=np.sum)
mydata_pivot2 = pd.pivot_table(mydata, index=['user','item'], values=['timestamp'], aggfunc=np.max)
mydata_pivot = pd.concat((mydata_pivot1, mydata_pivot2), axis=1)


# Normalized rate column based on th total number of users intraction
item_count = pd.DataFrame(mydata.groupby('user')['item'].count())
for index, row in mydata_pivot.iterrows():
    mydata_pivot.set_value(index,'rating', float(float(row['rating']) / float(item_count.item[index[0]])))


# Scale rate to 1-5
min_max_scaler = MinMaxScaler(copy=True, feature_range=(1,5))
mydata_pivot[['rating']] = min_max_scaler.fit_transform(mydata_pivot[['rating']].values)

# Convert pivot table to csv file format
mydata_pivot.to_csv('data_rating_timestamp_normalized.csv')

# Load the prepare dataset
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1,5), skip_lines=1)

os.system("cls")

custom_dataset_path = ('.\data_rating_timestamp_normalized.csv')

print("\n\nUsing: " + custom_dataset_path)
print("\nLoading data...")
data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)
print("\nOK")

print ('---------------------------------------------------------------')
################################################# Algorithm Selection #######################################                       
print('Select the model: \n\n 1-SVD \n 2-KNN \n 3-Cosine Similarity \n 4-Pearson Similarity')
algo_select = input("Enter number 1-4: ")
algo_select = int (algo_select)

if algo_select == 1:
    # Using the famous SVD algorithm to build the model 
    print('\nModel is SVD ')
    algo = SVD()
    
elif algo_select == 2:
    # Use KNNBasic algorithm to build the model 
    print('\nModel is KNN ')
    algo = KNNBasic()
    
elif algo_select == 3:
    # use Cosine similarity measure to estimate rating 
    print('\nModel is Cosine Similarity ')
    sim_options = {'name': 'cosine',
                   'user_based': False  # compute  similarities between items
                   }
    algo = KNNBasic(sim_options=sim_options)
    
else:
    # use Pearson similarity measure to estimate rating 
    print('\nModel is Pearson Similarity ')
    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0  # no shrinkage
                   }
    algo = KNNBasic(sim_options=sim_options)
    
print ('---------------------------------------------------------------')
############################################## Make train and test data ##################################
print('How making the train and test data: \n\n 1-All dataset for train and test \n 2-Random selection based on determined percentage \n 3-K-Fold cross validation with k=3')
split_select = input("Enter number 1-3: ")
split_select = int(split_select)

if split_select == 1:
    # Retrieve all trainset for train and test to descibes the performances of an algorithm
    print ('\nModel is made by all dataset...\n')
    trainset = data.build_full_trainset()
    testset =trainset.build_testset() 

elif split_select == 2:
    # Split a dataset into trainset and testset (random) 25% Test and &5% Train
    print ('\nTrain = 75%  and test = 25 % all data...\n')
    trainset, testset = train_test_split(data, train_size=0.75 , test_size=0.25 , random_state=42)
    
else :
    # Split to run k-fold cross-validation (k=3) - A basic cross-validation iterator
    print('\nk-fold cross-validation...\n')
    kf = KFold(n_splits=3)
    

############################################# Build the model and test the algorithm ################################
print ('---------------------------------------------------------------')


if split_select == 1 or split_select == 2:
    
    print ('---------------------------------------------------------------')
    algo.fit(trainset) # Train the algorithm on the trainset
    predictions = algo.test(testset) # Predict ratings for the testset
    print("\nAccuracy:\n ")
    accuracy.rmse(predictions , verbose=True) # Compute RMSE
    accuracy.mae(predictions, verbose=True)
    top_n = get_top_n(predictions, n=10)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
    
else :
    # We do this during a cross-validation procedure! 
    for i, (trainset_cv, testset_cv) in enumerate(kf.split(data)):
        print ('---------------------------------------------------------------')
        print('Fold number', i + 1)
        algo.fit(trainset_cv)
    
        print('On testset :',end = ' ')
        predictions = algo.test(testset_cv)
        accuracy.rmse(predictions, verbose=True)
        accuracy.mae(predictions, verbose=True)
    
        print('On trainset:', end = ' ')
        predictions = algo.test(trainset_cv.build_testset())
        accuracy.rmse(predictions, verbose=True)
        accuracy.mae(predictions, verbose=True)
            
        top_n = get_top_n(predictions, n=10)
    
        precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)
        print ('---------------------------------------------------------------')

# Precision and recall can then be averaged over all users
print ('---------------------------------------------------------------')
print('\nPrecision = ',sum(prec for prec in precisions.values()) / len(precisions))
print('Recall = ', sum(rec for rec in recalls.values()) / len(recalls),'\n')

# Get a prediction for specific users and items or...
print ('---------------------------------------------------------------')
uid = input("What is the user id for recommended items?(For example 12286) ")
uid = str(uid)  # raw user id (as in the ratings file). They are **strings**!
uid_items = top_n[uid]
if uid_items == []:
    print ('There is no user with this id...')    
else:
    print('Recommneder Items for user ', uid,' is: ' ,uid_items)

# Print the recommended items for all users
print ('---------------------------------------------------------------')
print ('Do you want to see recommended items for all user? (y/n)')
answer = input("y or n ? ")
if answer == 'y':
    print("Results:")
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])


# Run 5-fold cross-validation and print results
print ('---------------------------------------------------------------')
print ('Do you want to see 5-fold cross-validation result too? (y/n)')
answer = input("y or n ? ")
print('\nPlease Wait.....\n')
if answer == 'y':
    os.system("cls")
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
else:
    print ('---------------------------------------------------------------')
    print('Goodbye ......')
