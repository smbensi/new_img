# #====================================================
# # Class: LocationCollection
# # 
# # Class for Location Collection. 
# # Inherited form MogoCollection
# #====================================================

# from bson import ObjectId
# import pymongo
# from img_xtend.database.MongoCollection import *
# from img_xtend.database import Collections

# class VoiceCollection(MongoCollection):
#     def __init__(self) -> None:
#         MongoCollection.__init__(self, Collections.Robot)

#     def get_voice_attrs(self):
#         try:   
#             collection = self.db["robot"]
               
#             cursor = collection.find_one()
            
#             voice_attrs = cursor.get("voice","default")
            
#         except Exception as e:
#             self.logger.error(f"Error occured reading from: {self.CollectionName}, {e}")
#         return voice_attrs


#====================================================
# Class: FaceVectorsCollection
# 
# Class for Face Vectors Collection. 
# Inherited form MogoCollection
#====================================================

from img_xtend.database.MongoCollection import *
from img_xtend.database import Collections

class FaceVectorsCollection(MongoCollection):

    def __init__(self) -> None:
        MongoCollection.__init__(self, Collections.Faces)

    def AddOneFaceVector(self, dataSet):
        try:
            # first delete the record
            myquery = { "_id": dataSet['_id'] }
            self.Collection.delete_one(myquery)
            if len(dataSet["faceVector"]) > 0:
                if self.Collection.insert_one(dataSet):
                    self.logger.info(f"{self.CollectionName} Face Vector added for Id: {dataSet['_id']}")
                
        except Exception as e:
            self.logger.error(f"Error occured writing to: {self.CollectionName}, {e}")


    
    # this is primarily for testing, but could be used for multiple vectors I guess.
    def AddBatchFaceVectors(self, dataSet):
        for rec in dataSet:
            self.AddOneFaceVector(rec)


    def GetFaceVectors_old(self):
            rows = []
            try:
                cursor = self.Collection.aggregate([{'$lookup':{'from' : Collections.Users, 'localField' : '_id', 'foreignField': '_id', 'as': 'user'}}])
                for doc in cursor:
                    doc2 = {}
                    doc2['_id'] = doc['_id'] 
                    if doc['user']:
                        doc2['firstname'] = doc['user'][0]['firstname']
                    else:
                        continue
                    doc2['facevector'] = doc['faceVector']
                    rows.append(doc2)

            except Exception as e:
                self.logger.error(f"Error occured reading from: {self.CollectionName}, {e}")

            return rows

    def GetFaceVectors(self):
        return self.GetData()
