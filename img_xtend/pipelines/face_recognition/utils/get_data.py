import time
import json
import os
from typing import List, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from Database import UsersCollection as UC
from Database import FaceVectorsCollection as FVC

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
    
    names = []
    ids = []
    collections = []
    names_and_vec = {}
    names_and_ids = {}
    # embeddings = []
    np_emb = np.empty((1,512))
    
    df = pd.json_normalize(face_emb)
    cfg.ALL_FACE_VECTORS = df
    logger.debug(df)
    logger.debug(f"############### {df.columns=}")

    if os.getenv("DEBUGGING",False)=="True":
        df_to_file = df.to_pickle('/code/shared/db.pkl')
    return df
    
    # for face in face_emb:
    #     for i in enumerate(face[MAPPING.FACE_VECTORS.value]):
    #         ids.append(face.get(MAPPING.ID.value,"no ID"))
    #         names.append(face.get(MAPPING.NAME.value,"no name"))
    #         collections.append(face.get(MAPPING.COLLECTION.value,"no collection"))
            
    #     vecs = np.array(face[MAPPING.FACE_VECTORS.value]).astype(np.float64)
    #     if vecs.shape[0] == 0:
    #         continue
        
    #     np_emb = np.concatenate((np_emb,vecs),axis=0)
        
    #     names_and_vec[face[MAPPING.NAME.value]]=len(face[MAPPING.FACE_VECTORS.value])
    #     names_and_ids[face[MAPPING.NAME.value]]= face.get(MAPPING.ID.value,"no ID")
    
    # logger.info(f"** NAMES AND NB OF VECTORS IN DB *** {names_and_vec}")
    # logger.info(f"** NAMES AND CORRESPONDING ID *** {names_and_ids}")
    # embeddings = np_emb[1:]
    # return ids,names, embeddings, collections

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
    return ids,names, embeddings
