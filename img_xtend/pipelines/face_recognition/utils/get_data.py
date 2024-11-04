import time
import json
import os
from typing import List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

# from Database import UsersCollection as UC
# from Database import FaceVectorsCollection as FVC
from img_xtend.database import FaceVectorsCollection as FVC

from img_xtend.pipelines.face_recognition.configuration import config as cfg
from img_xtend.utils import LOGGER as logger

class MAPPING(Enum):
    FACE_VECTORS: np.array = "faceVector"
    ID: str = "_id"
    NAME: str = "name"
    COLLECTION: str = "collectionName"
    
def get_data_from_db() -> Tuple[List[str],List[str],List[np.ndarray],List[str]]: 
    # tmp = UC.UserCollection()
    tmp = FVC.FaceVectorsCollection()
    face_emb = tmp.GetFaceVectors()
    
    df = pd.json_normalize(face_emb)
    cfg.ALL_FACE_VECTORS = df
    logger.debug(df)
    logger.debug(f"############### {df.columns=}")

    if os.getenv("DEBUGGING",False)=="True":
        df_to_file = df.to_pickle('/code/shared/db.pkl')
    return df

def filter_face_vecs(column:str=MAPPING.COLLECTION.value,
                     values:List[str]=None)-> pd.DataFrame:
    df = cfg.ALL_FACE_VECTORS
    collections = list(df[MAPPING.COLLECTION.value].unique())
        
    logger.debug(f"Unique values {collections=} and {type(collections)=}")
    if values==None or values==[]:
        return df
    else:
        return df[df[column].isin(values)]

def get_data_from_json(json_file):
    file = json.load(open(json_file))
    names = []
    ids = []
    embeddings = []
    names.append("JohnyJohn" if not "name" in file else file["name"])
    names *=10
    ids.append("1234" if not "_id" in file else file["_id"] )
    ids *=10
    embeddings.append(np.array(file["faceVector"]).astype(np.float64))
    logger.debug(f"** NAMES IN DB *** {names}")
    embeddings = np.array(embeddings).reshape((-1,512),order='C')
    collections = ["user"]*len(names)
    
    json_db = {
                MAPPING.ID.value: ids,
                MAPPING.COLLECTION.value: collections,
                MAPPING.NAME.value: names,
                MAPPING.FACE_VECTORS.value: list(embeddings)
                
            }
    df = pd.DataFrame(json_db)
    cfg.ALL_FACE_VECTORS = df
    
    return df
