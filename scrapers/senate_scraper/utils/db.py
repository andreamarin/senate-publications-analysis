import os
import json
import pathlib
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi


def connect_mongo_db(db_name: str):
    """
    Parameters
    ----------
    db_name : str
        _description_

    Returns
    -------
    MongoClient
        client connected to mongo db
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    parent_path = current_path.parent.resolve()
    cert_file = f"{parent_path}/config/bot-cert.pem"

    uri = "mongodb+srv://senate-publication.at2rlna.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
    client = MongoClient(
        host=uri,
        tls=True,
        tlsCertificateKeyFile=cert_file,
        server_api=ServerApi('1')
    )
    return client[db_name]


def publication_exists(publication_id: str, table_name, conn) -> bool:
    """
    Check if the publication's data is in the db
    """

    table = conn[table_name]
    num_records = table.count_documents({"_id": publication_id})
    
    return num_records > 0


def save_publication_json(publication):
    """
    Save publication to json file
    """
    date_path = publication.date.strftime("year=%Y/month=%m/day=%d")
    save_path = f"{os.getcwd()}/data/{date_path}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    json_path = f"{save_path}/{publication.id}.json"
    with open(json_path, "w") as f:
        comm_dict = {k: v for k, v in publication.__dict__.items() if not k.startswith("_")}
        comm_dict["date"] = comm_dict["date"].isoformat()
        json.dump(comm_dict, f)


def save_publications(publications, table_name, conn):
    """
    Save publications

    Parameters
    ----------
    publication : _type_
        _description_
    table_name : _type_
        _description_
    conn : _type_
        _description_
    """
    insert_many = type(publications) == list
    table = conn[table_name]

    if insert_many:
        table.insert_many(publications)
    else:
        table.insert_one(publications)


def update_publications(publications, table_name, conn):
    """
    Save publications

    Parameters
    ----------
    publication : _type_
        _description_
    table_name : _type_
        _description_
    conn : _type_
        _description_
    """
    update_many = type(publications) == list
    table = conn[table_name]

    if update_many:
        updates = [
            UpdateOne(
                {"_id": p["_id"]},
                {"$set": {k: v for k, v in p.items() if k != "_id"}}
            ) 
            for p in publications
        ]
        table.bulk_write(updates)
    else:
        update_fields = {k: v for k, v in publications.items() if k != "_id"}
        table.update_one({"_id": publications["_id"]}, {"$set": update_fields})
