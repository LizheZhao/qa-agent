import os

from pymongo import MongoClient


def get_mongodb(db_name):
    host = os.getenv("MONGODB_HOST")
    port = os.getenv("MONGODB_PORT")
    user = os.getenv("MONGODB_USER")
    pwd = os.getenv("MONGODB_PWD")
    db = os.getenv("MONGODB_DB")
    mongo_uri = (
        f"mongodb://{user}:{pwd}@{host}:{port}/{db}?directConnection=true&authMechanism=SCRAM-SHA-256"
    )

    client = MongoClient(mongo_uri, uuidRepresentation="standard")
    db = client[db_name]

    return db


db = get_mongodb("ask-genome")
