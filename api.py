# from flask import Flask, jsonify
# from pymongo import MongoClient
# from bson import ObjectId
# from bson import json_util

# app = Flask(__name__)

# # Kết nối đến cơ sở dữ liệu MongoDB
# client = MongoClient("mongodb://localhost:27017")
# db = client["test"]
# collection = db["jobposts"]


# # Tạo API để lấy dữ liệu từ MongoDB
# @app.route("/data")
# def get_data():
#     data = []
#     for document in collection.find():
#         data.append(json_util.dumps(document))
#     return jsonify(data)


# if __name__ == "__main__":
#     app.run(port=8080)


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
                # "data": json.dumps(document, default=json_util.default),
            }
        )
    return jsonify(resultsFull)


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
        res_id = str(resumes_df["_id"][indices[0][t]])
        resume_doc = resumes_colle.find_one({"_id": ObjectId(res_id)})
        resultsFull.append(
            {
                "resId": res_id,
                "rs1": distances[0][t],
                "data": json.loads(json_util.dumps(resume_doc)),
            }
        )
    return jsonify(resultsFull)
