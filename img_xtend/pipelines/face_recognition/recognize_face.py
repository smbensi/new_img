import os
from typing import List
import time
import json

import numpy as np
import pandas as pd
import cv2

from img_xtend.utils import LOGGER, ROOT_PARENT
from img_xtend.pipelines.face_recognition.utils.get_data import (
    get_data_from_db,
    get_data_from_json,
    filter_face_vecs,
    MAPPING
)
from img_xtend.pipelines.face_recognition.utils.face_dataclasses import UnknownFace, KnownFace
from img_xtend.pipelines.face_recognition.configuration import config as cfg
from img_xtend.recognition.resnet import ResNetModel

class FaceRecognition():
    """de quoi j'ai besoin:
    - init 10
    - update: 1
        - recoit le frame et calcule les embeddings
        - envoit a une function qui update les listes 2
            - update les recognized 3
            - update les unrecognized 4
          et renvoi qui elle a vu dans le frame
        - en fonction de qui on voit dans la frame preparer le message MQTT 5
    
    fcts help:
        - delete multiple elements from a list 6
        - checker si une recognized face ressemble a une unrecognized 7
        - calcule la plus petite distance avec les recognized faces 8
        - check biggest 9
        
    """
    
    def __init__(self) -> None:
        
        self.faces_df = pd.DataFrame()
        self.recognition_model = ResNetModel()
        self.get_registered_faces()
        self.initialize_params()
        
    def get_registered_faces(self, source:str = cfg.DATA_SOURCE):
        """Get the data of face_vectors, name and collection from diffent sources possible"""
        if source == "DB":
            self.faces_df = get_data_from_db()
        elif source=="JSON":
            self.faces_df = get_data_from_json(f"{ROOT_PARENT}/shared/Jakes_photos/1234.json")
        else: 
            pass
        LOGGER.debug("*** DATA UPDATED ***")

    def initialize_params(self, collectionNames=None):
        self.dict_known_faces = {}  # keys: id, value: {"frames","collectionName","last_seen",}
        self.list_unknown_faces = []
        self.last_len_of_unreco_list = 0
        
        # self.published_id = ''
        self.published = False
        # self.face_timeout = 0
        self.mqtt_params()

        LOGGER.debug(f"{collectionNames=}")
        self.faces_df = filter_face_vecs(values=collectionNames)
        LOGGER.debug(f"{self.faces_df[['_id','name', 'collectionName']]=}")
        
        
        # for debugging
        self.previous_person = None
        self.previous_val = 10
        self.previous_min_dist_unknown = 10
        self.previous_min_unknown_index = None
        
    def mqtt_params(self):
        # for initialization
        self.last_time_published = 0
        self.index_unknown = 0
        self.msg_published = {"faces":[]}
        
        self.last_published_recognized_ids = []
        self.last_published_unrecognized_indexes = []
    
    def update_unknown_list(self, face_vector : np.ndarray, face_bbox: List[int]=None):
        """
        Update the list of unknown faces
        We have 3 options:
            1. Create a new person
            2. Update an existing one
            3. Delete a person that has not been seen since a long time (defined as cfg.NULL_TIMEOUT)

        Args:
            face_vector (np.ndarray): _description_
            face_bbox (List[int], optional): _description_. Defaults to None.

        Returns:
            unreco_ids: List[int] : all the indexes from the list of unknown persons recognized in the frame
        """
        
        # for debugging
        if self.last_len_of_unreco_list != len(self.list_unknown_faces):
            LOGGER.debug(f"length list unrecognized changed: BEFORE: {self.last_len_of_unreco_list} NOW {len(self.list_unknown_faces)}")
        self.last_len_of_unreco_list = len(self.list_unknown_faces)
        # end for debugging
        
        
        unreco_ids = []
        elements_to_delete = []
        add_new = False
        debug_min_dist = 100
        debug_index_min = -1
        
        # check which elements in the list we have to delete
        for i,unknown_face in enumerate(self.list_unknown_faces):
            if (time.time() - unknown_face.last_seen) > cfg.NULL_TIMEOUT :
                    elements_to_delete.append(i)
                    
        self._delete_elements_from_list(elements_to_delete, "unknown")
        
        if len(self.list_unknown_faces) == 0:
            self.list_unknown_faces.append(UnknownFace(face_vector=face_vector,
                                                    frames=1,
                                                    last_seen=time.time(),
                                                    bbox=face_bbox,
                                                    index = self.index_unknown))
            LOGGER.debug(f"Create new unknown face with id: {self.index_unknown}")
            self.index_unknown +=1
        
        else:
            dists_from_unknown_faces = []
            for i,elem in enumerate(self.list_unknown_faces):
                # dist = (face_vector - elem.face_vector).norm().item()
                dist = np.linalg.norm(face_vector - elem.face_vector)
                dists_from_unknown_faces.append(dist)
                
                if dist < debug_min_dist:
                    debug_min_dist = dist
                    debug_index_min = i
                    
                if dist < cfg.SIMILARITY_THRESHOLD_UNKNOWN:
                    if time.time() - elem.last_seen > cfg.NULL_TIMEOUT:
                        self.list_unknown_faces[i].frames = 1
                    else:
                        self.list_unknown_faces[i].frames += 1
                    self.list_unknown_faces[i].face_vector = face_vector 
                    self.list_unknown_faces[i].last_seen = time.time()
                    self.list_unknown_faces[i].bbox = face_bbox
                    self.list_unknown_faces[i].miss = 0
                    
                    unreco_ids.append(i)
                    # add_new = False
                    
                    nb_of_frames = self.list_unknown_faces[i].frames 
                    if (nb_of_frames <= 60 and nb_of_frames % 15 == 0) or (nb_of_frames>60 and nb_of_frames%100 == 0):
                        LOGGER.debug(f"index:{i:<5} distance:{dist:<5.2f} frames:{self.list_unknown_faces[i].frames}")
                    # break
                else:
                    self.list_unknown_faces[i].miss += 1
            
            
            if all(value > cfg.NEW_UNKNOWN_PERSON for value in dists_from_unknown_faces):
                try:     
                    self.list_unknown_faces.append(UnknownFace(face_vector=face_vector,
                                                            frames=1,
                                                            last_seen=time.time(),
                                                            bbox=face_bbox,
                                                            index=self.index_unknown
                                                            ))
                    self.index_unknown+=1
                    unreco_ids.append(len(self.list_unknown_faces)-1)
                except Exception as e:
                    LOGGER.debug(e)

            if debug_min_dist > cfg.SIMILARITY_THRESHOLD_UNKNOWN \
            and debug_min_dist <self.previous_min_dist_unknown \
            and debug_index_min != self.previous_min_unknown_index:
                LOGGER.debug(f"min dist:{debug_min_dist:.2f} from unknown index:{debug_index_min}")
                self.previous_min_dist_unknown = debug_min_dist
                self.previous_min_unknown_index = debug_index_min
        
        if len(unreco_ids)> 1:
            LOGGER.debug(f"ONE VECTOR IS ADAPTED TO MULTIPLE UNRECO IDS: {unreco_ids=}")
            # merge same faces
            self._delete_elements_from_list(unreco_ids[1:], "unknown"
                                            )
        return [self.list_unknown_faces[unreco_ids[0]].index] if len(unreco_ids)>0 else unreco_ids       
                     
    def get_minimum_distance(self,face_vec):
        if not isinstance(face_vec, np.ndarray):
            face_vec = face_vec.numpy()
        try:
            dist = [min(np.linalg.norm((np.array(row[MAPPING.FACE_VECTORS.value],dtype=np.float64) - face_vec),
                               axis=1))
                    if np.array(row[MAPPING.FACE_VECTORS.value],dtype=np.float64).shape[0]>0 else 10
                    for index,row in self.faces_df.iterrows()] 
                    
            val, idx = min((val, idx) for (idx, val) in enumerate(dist))
        except ValueError as exeption:
            LOGGER.debug(f"***EXCEPTION: {Exception}")
            return 10,0
        return val,idx
    
    def _delete_elements_from_list(self, indexes:List[int], list_type: str):
        # LOGGER.debug(f"indexes to remove {indexes} in {list_type}")
        for i in sorted(list(set(indexes)), reverse=True):
            try:
                if list_type == "unknown":
                    del self.list_unknown_faces[i]
                    # LOGGER.debug(f"AFTER_REMOVE : {self.list_unknown_faces}")  
                elif list_type == "known":
                    del self.dict_known_faces[i]
                    # LOGGER.debug(f"AFTER_REMOVE : {self.dict_known_faces}")
            except IndexError as e:
                list_of_faces = self.list_unknown_faces if list_type=="unknown" else self.dict_known_faces
                LOGGER.error(f"ERROR can remove {i} index from {list_of_faces} ")
            
    def update_recognized_list(self, id_recognized: str, collectionName:str, biggest: bool, face_bbox = None):
        # delete old faces recognized
        try:
            faces_to_delete = []
            for el in self.dict_known_faces:
                if (time.time() - self.dict_known_faces[el].last_seen) > cfg.NULL_TIMEOUT:
                    faces_to_delete.append(el)
            
            self._delete_elements_from_list(faces_to_delete, "known")
        except Exception as e:
            LOGGER.debug(f"Error in {update_recognized_list.__name__}: {e}")

        if id_recognized in self.dict_known_faces:
            if time.time() - self.dict_known_faces[id_recognized].last_seen < cfg.NULL_TIMEOUT:
                self.dict_known_faces[id_recognized].frames += 1
            else:
                self.dict_known_faces[id_recognized].frames = 1
                
            self.dict_known_faces[id_recognized].bbox = face_bbox
            self.dict_known_faces[id_recognized].last_seen = time.time()
        
        else:
            self.dict_known_faces[id_recognized] = KnownFace(_id = id_recognized,
                                                             collectionName=collectionName,
                                                             frames=1,
                                                             last_seen=time.time(),
                                                             bbox=face_bbox)
           
    def check_biggest(self,ids : List[str]):
        if len(ids) > 1:
            for id in ids[1:]:
                if self.dict_known_faces[id].frames <= cfg.FRAMES_BEFORE_RECOGNITION and \
                    self.dict_known_faces[id].frames >= self.dict_known_faces[ids[0]].frames:
                    self.dict_known_faces[id].frames = 0
                     
    def check_if_recognized_in_unrecognized(self, recognized_vec):
        keys_to_delete = []
        
        for i,unknown_face in enumerate(self.list_unknown_faces):
            if (unknown_face.face_vector - recognized_vec).norm().item() < cfg.SIMILARITY_THRESHOLD:
                LOGGER.debug(f"********************DELETING the Key {i}")
                keys_to_delete.append(i)
        
        self._delete_elements_from_list(keys_to_delete, 'unknown')
    
    
    def recognized_faces_to_publish(self, ids:List[str]=None):
        """

        Args:
            ids: List[str] : ids from database of recognized faces in the frame
        """
        
        # remove old recognized faces and from other collections
        ids_to_remove = []
        for i,face in enumerate(self.last_published_recognized_ids):
            if (time.time() - self.dict_known_faces[face].last_seen) > cfg.NULL_TIMEOUT: # or \
            # self.dict_known_faces[face].collectionName not in cfg.COLLECTIONS_TO_RECOGNIZE:
                ids_to_remove.append(face)
                
        ids_to_publish = [el for el in ids 
                        if self.dict_known_faces[el].frames == cfg.FRAMES_BEFORE_RECOGNITION]
        
        ids_to_publish.extend([id for id in self.last_published_recognized_ids if id not in ids_to_remove])
        # LOGGER.debug(f"{self.last_published_recognized_ids=} , {ids=} ,  {list(set(ids_to_publish))=}")
        return list(set(ids_to_publish))
    
    def unrecognized_faces_to_publish(self, unrecognized_ids):
        unknowns_indexes_to_publish = []
        list_of_faces_to_publish = []
        for face in  self.list_unknown_faces:
            if face.index in unrecognized_ids and face.frames == cfg.UNKNOWN_FRAMES or \
            (face.index in self.last_published_unrecognized_indexes and \
            (time.time() - face.last_seen) < cfg.NULL_TIMEOUT):
                unknowns_indexes_to_publish.append(face.index)
                json_to_publish = {}
                json_to_publish["unknownId"] = face.index
                if True: #cfg.SEND_REC:
                    bbox = face.bbox
                    bbox_dict = {"x":int(bbox[0]),"y":int(bbox[1]),"w":int(bbox[2]),"h":int(bbox[3]),}
                    json_to_publish["posn"] = bbox_dict
                list_of_faces_to_publish.append(json_to_publish)
        
        
        return unknowns_indexes_to_publish, list_of_faces_to_publish

    def publish_mqtt(self,ids=None,unrecognized_ids=None):
        
        '''
        This function handles the publishing of MQTT messages
        Args:
            ids: List[str(_ids from the DB)] : index of recognized faces in the frame
            unrecognized_ids:List[int] : index of unrecognized faces in the frame
        '''
        # begin with recognized face dict then unrecognized faces
        ids_to_publish = []
        unknowns_ids_to_publish = []
        unknowns_indexes_to_publish = []
        # LOGGER.debug(f"{ids=}  and {unrecognized_ids=} and {len(self.list_unknown_faces)=}")
        
        
        def info_recognized(el):
            json_to_publish = {}
            json_to_publish = {"_id":el, 
                                    "collectionName": self.dict_known_faces[el].collectionName
                                    }
            if True: #cfg.SEND_REC:
                    bbox = self.dict_known_faces[el].bbox
                    bbox_dict = {"x":int(bbox[0]),"y":int(bbox[1]),"w":int(bbox[2]),"h":int(bbox[3]),}
                    LOGGER.debug(f"{bbox_dict=}")
                    json_to_publish["posn"] = bbox_dict
            return json_to_publish
        
        self.msg_published = {"faces":[]}
        
        # LOGGER.debug(f"{unrecognized_ids=} ,{self.list_unknown_faces=}")
        
        recognized_ids_to_publish = self.recognized_faces_to_publish(ids)
        unknowns_ids_to_publish, unknowns_faces_to_publish = self.unrecognized_faces_to_publish(unrecognized_ids)
        # LOGGER.debug(f"{recognized_ids_to_publish=}")
        # LOGGER.debug(f"{unknowns_ids_to_publish=}")
        
        if (cfg.MSG_SPEED<0  and (self.last_published_recognized_ids != recognized_ids_to_publish or \
            self.last_published_unrecognized_indexes != unknowns_ids_to_publish)) or \
                (cfg.MSG_SPEED > 0 and (time.time()-self.last_time_published)>cfg.MSG_SPEED):
            for el in recognized_ids_to_publish:            
                self.msg_published["faces"].append(info_recognized(el))
            self.msg_published["faces"].extend(unknowns_faces_to_publish)
            if len(self.msg_published["faces"]) > 1:
                self.msg_published["faces"] = sorted(self.msg_published["faces"],key=lambda x:x["posn"]["w"]*x["posn"]["h"],reverse=True)
            msg = json.dumps(self.msg_published)
            LOGGER.debug(f"MESSAGE TO PUBLISH MQTT {msg=} ")
            LOGGER.debug(f"{self.dict_known_faces=} \n{self.list_unknown_faces=}")
            if cfg.FACE_CLIENT is not None:
                cfg.FACE_CLIENT.publish(
                            cfg.TOPICS_TO_BRAIN["FACE_RECOGNITION"], msg)
                
            self.last_time_published = time.time()  
            self.last_published_recognized_ids = recognized_ids_to_publish    
            self.last_published_unrecognized_indexes = unknowns_ids_to_publish                
            
    def check_if_recognized(self, face_info):
        
        id_recognized = None
        name_recognized = None
        val = None
        
        face_bbox = face_info['posn']
        face_embedding = face_info["face_embedding"]
        
        if len(self.faces_df) > 0:
            val,idx = self.get_minimum_distance(face_embedding)
            print(f"{val=} and person {self.faces_df.name.iloc[idx]}")
            if val < cfg.SIMILARITY_THRESHOLD:
                try:
                    id_recognized = self.faces_df[MAPPING.ID.value].iloc[idx]
                    name_recognized = self.faces_df.name.iloc[idx]
                    collectionName = self.faces_df[MAPPING.COLLECTION.value].iloc[idx]
                    
                    self.update_recognized_list(id_recognized,collectionName,biggest=False, face_bbox=face_bbox)
                    self.check_if_recognized_in_unrecognized(face_embedding)
                except Exception as e:
                        LOGGER.debug(f"Error in updating know faces: {e}")
                        
        return id_recognized, name_recognized, val
    
    def mark_unknown_missed(self):
        for i,elem in enumerate(self.list_unknown_faces):
            self.list_unknown_faces[i].miss += 1
            
    def update(self,img,pose_estimation_results):
        """
        Control all the pipeline and the steps:
            1. Get the faces from the pose 
            2. compute the vectors
            for each face:
                3. find if it's a recognized face
                4. find if it's an unrecognized face
            5. check if we need to publish MQTT

        Args:
            img (_type_): _description_
        """
        face_vecs_in_frame = []
        for pose_result in pose_estimation_results:
            from img_xtend.pipelines.face_recognition.utils.find_face import get_face_from_pose
            from img_xtend.pipelines.face_recognition.utils.visualization import show_pose
            img_with_pose = show_pose(img, pose_result)
            cv2.imwrite(f'img.jpg',img_with_pose)
            face_info = get_face_from_pose(pose_result, img)
            if "face" in face_info:
                cv2.imwrite(f'face.jpg',face_info["face"])
                face_info["face_embedding"] = self.recognition_model(face_info["face"])
            else:
                continue        
            face_vecs_in_frame.append(face_info)
        # face_vecs_in_frame = self.get_face_vectors(img)
        ids = []
        names_reco = []
        vals = []
        unreco_ids = []
        first_recognized = True
        indexes_vectors_recognized = []
        
        # check if we have recognized faces:
        for i, face_vector in enumerate(face_vecs_in_frame):
            id_recognized, name_recognized, val = self.check_if_recognized(face_vector)
            if id_recognized is not None:
                indexes_vectors_recognized.append(i)
                ids.append(id_recognized)
                names_reco.append(name_recognized) if name_recognized is not None else LOGGER.debug("error name recognized")
                vals.append(val) if val is not None else LOGGER.debug("error val")
        if ids:
                cond = [self.dict_known_faces[id].frames<= cfg.FRAMES_BEFORE_RECOGNITION or (self.dict_known_faces[id].frames> cfg.FRAMES_BEFORE_RECOGNITION and self.dict_known_faces[id].frames % 20 == 0) for id in ids ]
                if any(cond):
                    LOGGER.debug(f"names in the frame: {[(name,self.dict_known_faces[id].frames) for name,id in zip(names_reco, ids)]} with similarity= {[round(val,2) for val in vals]}")
                self.check_biggest(ids)
                
        # delete face vectors that has been recognized
        for index in sorted(list(set(indexes_vectors_recognized)), reverse=True):
            del face_vecs_in_frame[index]
        

        # check if we have unrecognized faces
        if len(face_vecs_in_frame)==0:
            self.mark_unknown_missed()
        else:    
            for i, face_info in enumerate(face_vecs_in_frame):
                face_bbox = face_info["posn"]
                face_embedding = face_info["face_embedding"]
                unreco_ids.extend(self.update_unknown_list(face_embedding,face_bbox=face_bbox))

        try:
            self.publish_mqtt(ids, unreco_ids)
        except KeyError as e:
            LOGGER.debug(f"Error in publish_mqtt: {e}")
            LOGGER.debug(f"dict known faces: {self.dict_known_faces}")
            
        
        # if self.published and (time.time() - self.face_timeout) >= cfg.NULL_TIMEOUT:
        #     LOGGER.debug("PUBLISH EMPTY HERE")
        #     cfg.FACE_CLIENT.publish(cfg.TOPICS_TO_BRAIN["FACE_RECOGNITION"],json.dumps({"face":[]}))
        #     self.initialize_params(collectionNames=cfg.COLLECTION)
