# import pandas as pd
# from pymongo import MongoClient

# # connect to MongoDB
# client = MongoClient(
#     "mongodb+srv://PhuongB1807662:maGL5yKdyHOOq3FC@smartjob.yutgdjn.mongodb.net/"
# )
# db = client["test"]
# jobpost_collection = db["jobposts"]
# resume_collection = db["resumes"]

# # load jobposts and resumes into pandas dataframes
# jobposts_df = pd.DataFrame(list(jobpost_collection.find()))
# resumes_df = pd.DataFrame(list(resume_collection.find()))

# # write dataframes to Excel files
# jobposts_df.to_excel("jobposts.xlsx", index=False)
# resumes_df.to_excel("resumes.xlsx", index=False)

## Phan2

import pandas as pd
from pymongo import MongoClient

# connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["job"]
jobpost_collection = db["jobposts"]
resume_collection = db["resumes"]

# load jobposts and resumes into pandas dataframes
jobposts_df = pd.DataFrame(list(jobpost_collection.find()))
resumes_df = pd.DataFrame(list(resume_collection.find()))

# merge title, candidateRequiredText, and descriptionText into one field
jobposts_df["combined_text"] = jobposts_df.apply(
    lambda x: f'{x["title"]}. {x["requiredText"]}. {x["descriptionText"]}',
    axis=1,
)

# select id and combined_text fields
jobposts_df = jobposts_df[["_id", "combined_text"]]

# write jobposts dataframe to Excel file
jobposts_df.to_excel("jobposts.xlsx", index=False)


# merge experience and skills into one field
resumes_df["combined_text"] = resumes_df.apply(
    lambda x: f'{x["experience"]}. {x["skills"]}. {x["title"]}', axis=1
)

# select id and combined_text fields
resumes_df = resumes_df[["_id", "combined_text"]]

# write resumes dataframe to Excel file
resumes_df.to_excel("resumes.xlsx", index=False)

## phan3

# import pandas as pd
# import re
# import numpy as np
# import nltk

# nltk.download("stopwords")
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from string import punctuation
# from nltk.corpus import wordnet as wn
# from nltk.stem import WordNetLemmatizer
# from nltk.probability import FreqDist
# from heapq import nlargest
# from collections import defaultdict
# import pandas as pd
# from nltk.collocations import *

# # load jobposts and resumes from Excel files into pandas dataframes
# jobposts_df = pd.read_excel("jobposts.xlsx")
# resumes_df = pd.read_excel("resumes.xlsx")


# # define a function to preprocess text
# def preprocess_text(text):
#     # remove non-alphanumeric characters
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     # convert text to lowercase
#     text = text.lower()
#     # remove stop words
#     stop_words = stopwords.words("english")
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     # return preprocessed text
#     return text


# # apply text preprocessing to combined_text field in jobposts dataframe
# jobposts_df["preprocessed_text"] = jobposts_df["combined_text"].apply(preprocess_text)

# # apply text preprocessing to combined_text field in resumes dataframe
# resumes_df["preprocessed_text"] = resumes_df["combined_text"].apply(preprocess_text)

# # create TfidfVectorizer object and fit it to preprocessed text data
# tfidf = TfidfVectorizer()
# tfidf.fit(jobposts_df["preprocessed_text"])

# # transform preprocessed text data into TF-IDF vectors
# jobposts_tfidf = tfidf.transform(jobposts_df["preprocessed_text"])
# resumes_tfidf = tfidf.transform(resumes_df["preprocessed_text"])

# # print preprocessed jobposts dataframe
# print(jobposts_df[["_id", "preprocessed_text"]])

# # print preprocessed resumes dataframe
# print(resumes_df[["_id", "preprocessed_text"]])

# # print the shape of the resulting TF-IDF vectors
# print(jobposts_tfidf.shape)
# print(resumes_tfidf.shape)


# import pandas as pd
# import re
# import numpy as np
# import nltk

# nltk.download("stopwords")
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from string import punctuation
# from nltk.corpus import wordnet as wn
# from nltk.stem import WordNetLemmatizer
# from nltk.probability import FreqDist
# from heapq import nlargest
# from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
# import pandas as pd
# from nltk.collocations import *

# # load jobposts and resumes from Excel files into pandas dataframes
# jobposts_df = pd.read_excel("jobposts.xlsx")
# resumes_df = pd.read_excel("resumes.xlsx")


# # define a function to preprocess text
# def preprocess_text(text):
#     # remove non-alphanumeric characters
#     text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
#     # convert text to lowercase
#     text = text.lower()
#     # remove stop words
#     stop_words = stopwords.words("english")
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     # return preprocessed text
#     return text


# # apply text preprocessing to combined_text field in jobposts dataframe
# jobposts_df["preprocessed_text"] = jobposts_df["combined_text"].apply(preprocess_text)

# # apply text preprocessing to combined_text field in resumes dataframe
# resumes_df["preprocessed_text"] = resumes_df["combined_text"].apply(preprocess_text)

# # create TfidfVectorizer object and fit it to preprocessed text data
# tfidf = TfidfVectorizer()
# tfidf.fit(jobposts_df["preprocessed_text"])

# # transform preprocessed text data into TF-IDF vectors
# jobposts_tfidf = tfidf.transform(jobposts_df["preprocessed_text"])
# resumes_tfidf = tfidf.transform(resumes_df["preprocessed_text"])

# # train the KNN model with job posts data
# knn = NearestNeighbors(n_neighbors=10, metric="cosine")
# knn.fit(jobposts_tfidf)

# # find the top k most similar resumes to each job post
# for i in range(len(jobposts_df)):
#     distances, indices = knn.kneighbors(jobposts_tfidf[i])
#     print("Job Post {}:".format(jobposts_df["_id"][i]))
#     for j in range(len(indices[0])):
#         print(
#             "  Resume {}: Distance {}".format(
#                 resumes_df["_id"][indices[0][j]], distances[0][j]
#             )
#         )

# for i in range(len(resumes_df)):
#     distances, indices = knn.kneighbors(resumes_tfidf[i])
#     print("Resume {}:".format(resumes_df["_id"][i]))
#     for j in range(len(indices[0])):
#         print(
#             "  Job Post {}: Distance {}".format(
#                 jobposts_df["_id"][indices[0][j]], distances[0][j]
#             )
#         )
