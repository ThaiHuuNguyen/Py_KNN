from flask import Flask, jsonify
from pymongo import MongoClient
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# initialize Flask app
app = Flask(__name__)

# set up MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["test"]
jobposts_col = db["jobposts"]
resumes_col = db["resumes"]

# load jobposts and resumes from MongoDB collections into pandas dataframes
jobposts_df = pd.DataFrame(list(jobposts_col.find()))
resumes_df = pd.DataFrame(list(resumes_col.find()))

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
knn = NearestNeighbors(n_neighbors=44, metric="cosine")
knn.fit(jobposts_tfidf)


# test the KNN model
jobtitle_query = "software engineer"
jobtitle_query_preprocessed = preprocess_text(jobtitle_query)
jobtitle_query_tfidf = tfidf.transform([jobtitle_query_preprocessed])
distances, indices = knn.kneighbors(jobtitle_query_tfidf)

print("Top 5 most similar job posts to the query:")


# define a route to handle job post search requests
@app.route("/jobsearch/<jobtitle>")
def job_search(jobtitle):
    # preprocess job title query
    jobtitle_preprocessed = preprocess_text(jobtitle)
    # transform preprocessed job title query into TF-IDF vector
    jobtitle_tfidf = tfidf.transform([jobtitle_preprocessed])
    # find the top k most similar job posts to the job title query
    distances, indices = knn.kneighbors(jobtitle_tfidf)
    print("Indices: ", indices)
    print("Distances: ", distances)
    # create a list to store search results
    results = []
    # loop through the most similar job posts
    for i in range(len(indices[0])):
        # get the job post id and distance from the query
        job_post_id = jobposts_df["_id"][indices[0][i]]
        # job_post_title = jobposts_df["title"][indices[0][i]]
        # print(f"Job post id: {job_post_id}")
        distance = distances[0][i]
        # get the job post from the MongoDB collection
        job_post = jobposts_col.find_one({"_id": job_post_id})
        # get the corresponding resume id from the resumes_df dataframe
        resume_id = resumes_df["_id"][indices[0][i]]
        # add the job post, resume id, and distance to the search results
        results.append(
            {
                "id": job_post_id,
                "resume_id": resume_id,
                "job_post": job_post,
                "distance": distance,
            }
        )
    # return the search results as a JSON object
    return jsonify(results)


# run the app
if __name__ == "__main__":
    app.run(debug=True)
