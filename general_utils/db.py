import os
import json
import pathlib
from typing import Union
from pymongo import MongoClient, UpdateOne
from pymongo.server_api import ServerApi


def connect_mongo_db(db_name: str) -> MongoClient:
    """
    Parameters
    ----------
    db_name : str
        name of the database to connect to

    Returns
    -------
    MongoClient
        client connected to mongo db
    """
    current_path = pathlib.Path(__file__).parent.resolve()
    parent_path = current_path.parent.resolve()
    cert_file = f"{parent_path}/config/bot-cert.pem"

    uri = "mongodb+srv://senate-publication.at2rlna.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"  # noqa: E501
    client = MongoClient(
        host=uri, tls=True, tlsCertificateKeyFile=cert_file, server_api=ServerApi("1")
    )
    return client[db_name]


def record_exists(record_id: str, table_name, conn) -> bool:
    """
    Check if the publication's data is in the db
    """

    table = conn[table_name]
    num_records = table.count_documents({"_id": record_id})

    return num_records > 0


def save_record_json(publication):
    """
    Save record to json file
    """
    date_path = publication.date.strftime("year=%Y/month=%m/day=%d")
    save_path = f"{os.getcwd()}/data/{date_path}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    json_path = f"{save_path}/{publication.id}.json"
    with open(json_path, "w") as f:
        comm_dict = {
            k: v for k, v in publication.__dict__.items() if not k.startswith("_")
        }
        comm_dict["date"] = comm_dict["date"].isoformat()
        json.dump(comm_dict, f)


def insert_records(records: Union[list, dict], table_name: str, conn):
    """
    Insert records into the given table

    Parameters
    ----------
    publications : Union[list, dict]
        List of records as dictionary or single record
    table_name : str
        Name of the table (collection) where records will be inserted
    conn :
        Connection to mongo db
    """
    insert_many = type(records) is list
    table = conn[table_name]

    if insert_many:
        table.insert_many(records)
    else:
        table.insert_one(records)


def build_update_statement(
    record: dict, dict_columns: list, array_columns: dict
) -> Union[list, dict]:
    """
    Build update statement for all the columns of the given record

    Parameters
    ----------
    record : dict
        record with fields to update
    dict_columns : list
        name of the dict columns in the table
    array_columns : dict
        dictionary with info on array columns in the table
            key - name of the column
            value - name of the column that can be used as id of the array records

    Returns
    -------
    Union[list, dict]
        statement to update the current record in the db
            * list - update with aggregation pipeline if `array_columns` were provided
            * dict - single $set statement if no `array_columns` were provided
    """
    record_columns = set(record.keys()) - {"_id"}

    # get columns in this record
    record_dict_columns = record_columns.intersection(dict_columns)
    record_array_columns = record_columns.intersection(array_columns.keys())
    record_normal_columns = record_columns - record_dict_columns - record_array_columns

    # add normal updates (replace the whole content of the field)
    update_fields = {col: record[col] for col in record_normal_columns}

    # add updates for all the fields in the dict columns
    update_fields.update(
        {
            f"{col}.{subcol}": record[col][subcol]
            for col in record_dict_columns
            for subcol in record[col]
        }
    )

    if record_array_columns:
        # create aggregation pipeline, first stage is to update normal and dict fields
        update_statement = [{"$set": update_fields}]

        for col in record_array_columns:
            array_id = array_columns[col]

            # add update stage for each item in the array column
            for array_item in record[col]:
                update_statement.append(
                    {
                        "$set": {
                            col: {
                                "$map": {
                                    "input": f"${col}",
                                    "in": {
                                        "$cond": {
                                            "if": {
                                                # update if id matches
                                                "$eq": [
                                                    f"$$this.{array_id}",
                                                    array_item[array_id],
                                                ]
                                            },
                                            "then": {
                                                # merge old version with new one
                                                "$mergeObjects": ["$$this", array_item]
                                            },
                                            "else": "$$this",
                                        }
                                    },
                                }
                            }
                        }
                    }
                )

            # add concatArrays stage to add new items
            update_statement.append(
                {
                    "$set": {
                        col: {
                            "$concatArrays": [
                                f"${col}",
                                {
                                    "$filter": {
                                        "input": record[col],
                                        # filter items that already exists in the db
                                        "cond": {
                                            "$not": {
                                                "$in": [
                                                    f"$$this.{array_id}",
                                                    f"${col}.{array_id}",
                                                ]
                                            }
                                        },
                                    }
                                },
                            ]
                        }
                    }
                }
            )
    else:
        update_statement = {"$set": update_fields}

    return update_statement


def update_records(
    records: Union[list, dict],
    table_name: str,
    conn,
    dict_columns: list = [],
    array_columns: dict = dict(),
):
    """
    Update the given columns of the records in the db

    Parameters
    ----------
    records : Union[list, dict]
        list of records to update or dict of single record to update
    table_name : str
        name of the target table
    conn : _type_
        connection to the mongo db
    dict_columns : list, optional
        name of the columns that contain a dict, the update will be done at a subfield level, by default []
    array_columns : dict, optional
        dictionary with the columns that contain arrays of dictionaries, the update will be done at item level
            - key: name of the column
            - value: name of the field in the array items that can be used as id
        , by default dict()
    """
    update_many = type(records) is list
    table = conn[table_name]

    if update_many:
        updates = [
            UpdateOne(
                filter={"_id": r["_id"]},
                update=build_update_statement(r, dict_columns, array_columns),
            )
            for r in records
        ]
        table.bulk_write(updates)
    else:
        table.update_one(
            {"_id": records["_id"]},
            build_update_statement(records, dict_columns, array_columns),
        )


def batch_update_records(
    records: list,
    table_name: str,
    conn: MongoClient,
    batch_size: int = 2000,
    dict_columns: list = [],
    array_columns: dict = dict(),
):
    """
    Update all the records in the list in batches of `batch_size`

    Parameters
    ----------
    records : list
        list of records to update
    table_name : str
        name of the target table
    conn : MongoClient
        connection to the mongo db
    batch_size : int, optional
        amount of records to update in each batch, by default 2000
    dict_columns : list, optional
        name of the columns that contain a dict, by default []
    array_columns : dict, optional
        dictionary with the columns that contain arrays of dictionaries and the corresponding field that can be used
        as the items' id, by default dict()
    """
    num_publications = len(records)
    for start in range(0, num_publications, batch_size):
        batch_publications = records[start:start + batch_size]
        update_records(
            batch_publications, table_name, conn, dict_columns, array_columns
        )
