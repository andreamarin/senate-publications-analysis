import os
import json
from datetime import datetime


def publication_exists(publication_id: str, publication_date: datetime) -> bool:
    """
    Check if the json with the publication's data exists
    """
    date_path = publication_date.strftime("year=%Y/month=%m/day=%d")
    publication_path = f"{os.getcwd()}/data/{date_path}/{publication_id}.json"

    return os.path.exists(publication_path)


def save_publication(publication):
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
