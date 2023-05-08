from flask import Flask, jsonify
from pymongo import MongoClient
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from bson import ObjectId, json_util
import json

# initialize Flask app
app = Flask(__name__)

# set up MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["job"]
jobposts_colle = db["jobposts"]
resumes_colle = db["resumes"]

# load jobposts and resumes from MongoDB collections into pandas dataframes
#

jobposts_df = pd.read_excel("jobposts.xlsx")
resumes_df = pd.read_excel("resumes.xlsx")


# define a function to preprocess text
def preprocess_text(text):
    # remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # convert text to lowercase
    text = text.lower()
    # remove stop words
    stop_words = stopwords.words("english")
    text = " ".join([word for word in text.split() if word not in stop_words])
    # return preprocessed text
    return text


# apply text preprocessing to combined_text field in jobposts dataframe
jobposts_df["preprocessed_text"] = jobposts_df["combined_text"].apply(preprocess_text)

# apply text preprocessing to combined_text field in resumes dataframe
resumes_df["preprocessed_text"] = resumes_df["combined_text"].apply(preprocess_text)

# create TfidfVectorizer object and fit it to preprocessed text data
tfidf = TfidfVectorizer()
tfidf.fit(jobposts_df["preprocessed_text"])

# transform preprocessed text data into TF-IDF vectors
jobposts_tfidf = tfidf.transform(jobposts_df["preprocessed_text"])
resumes_tfidf = tfidf.transform(resumes_df["preprocessed_text"])

print("Shape of jobposts_tfidf:", jobposts_tfidf.shape)
print("Shape of resumes_tfidf:", resumes_tfidf.shape)

# train the KNN model with job posts data
knn = NearestNeighbors(n_neighbors=38, metric="cosine")
knn.fit(jobposts_tfidf)


@app.route("/findResForJob/<id>")
def findResForJob(id):
    results = []
    resultsFull = []

    row = jobposts_df[jobposts_df["_id"] == id]

    distances, indices = knn.kneighbors(jobposts_tfidf[row.index[0]])

    for j in range(len(indices[0])):
        results.append(
            {
                "resId": resumes_df["_id"][indices[0][j]],
                "rs": distances[0][j],
            }
        )
    for t in range(len(results)):
        document = jobposts_colle.find_one({"_id": ObjectId(id)})

        print("-----------------------------------------------------", document)
        resultsFull.append(
            {
                "resId": resumes_df["_id"][indices[0][t]],
                "rs1": distances[0][t],
                "data": json.dumps(document, default=json_util.default),
            }
        )
    return jsonify(resultsFull)


@app.route("/findJobForCv/<id>")
def findJobForCv(id):
    results = []
    resultsFull = []

    row = resumes_df[resumes_df["_id"] == id]

    print(type(resumes_tfidf))
    print((resumes_tfidf))
    print("testttt", resumes_tfidf[row.index[0]])
    distances, indices = knn.kneighbors(resumes_tfidf[row.index[0]])

    for j in range(len(indices[0])):
        results.append(
            {
                "jobId": jobposts_df["_id"][indices[0][j]],
                "rs1": distances[0][j],
            }
        )
    for t in range(len(results)):
        document = resumes_colle.find_one({"_id": ObjectId(id)})

        print("-----------------------------------------------------", document)
        resultsFull.append(
            {
                "resId": resumes_df["_id"][indices[0][t]],
                "rs": distances[0][t],
                "data": json.dumps(document, default=json_util.default),
            }
        )
    return jsonify(resultsFull)


# run the app
if __name__ == "__main__":
    app.run(port=8080)
