#====================================================
# Class: MongoCollection
# 
# Class for all MongoDB collections
#====================================================
from uuid import getnode
from img_xtend.database.MongoDB import *
# import xtendlog
from img_xtend.utils import LOGGER

class MongoCollection:

    def __init__(self, CollectionName) -> None:
        self.db = MongoDB().get_db_instance()
        self.CollectionName = CollectionName
        self.Collection = self.db[CollectionName]
        # self.logger = xtendlog.loggerclass.get()
        self.logger = LOGGER
        
    def GetCollection(self):
        return self.CollectionName

    def AddData(self, dataSet):
        try:
            self.Collection.delete_many({})
            if self.Collection.insert_many(dataSet):
                self.logger.info(f"{self.CollectionName} Collection built")
                
        except Exception as e:
            self.logger.error(f"Error occured writing to: {self.CollectionName}, {e}")


    # Private method that does the addition. It returns the _id if succesful else None
    # return: _id
    def __AddOneDocument(self, dataSet, bDeleteAll=True):
        _id = None
        try:
            if (bDeleteAll):
                self.Collection.delete_many({})
    
            res = self.Collection.insert_one(dataSet)
            _id = str(res.inserted_id)
            self.logger.info(f"{self.CollectionName} Collection record added: {dataSet}")
                
        except Exception as e:
            self.logger.error(f"Error occured writing to: {self.CollectionName}, {e}")

        return _id

    # AddOneDocument cleans out the collection before adding a single document
    def AddOneDocument(self, dataSet):
        self.__AddOneDocument(dataSet, True)


    # InsertNewRecord adds a new record to an existing collection 
    # Return: _id
    def InsertNewRecord(self, dataSet):
        _id = self.__AddOneDocument(dataSet, False)
        return _id


    def GetData(self):
        rows = []
        try:
            cursor = self.Collection.find()
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                rows.append(doc)

            return rows

        except Exception as e:
            self.logger.error(f"Error occured while reading from: {self.CollectionName}, {e}")
        
        return rows


    def GetOneDocument(self, with_Id = 0, srch_id = None):
        row = {}
        attrs = {}
        if (with_Id == 0):
            attrs = {"_id" : 0 }
        Qry = {}
        if (srch_id is not None):
            Qry = {'_id' : srch_id}

        try:
            row = self.Collection.find_one(Qry, attrs)

        except Exception as e:
            self.logger.error(f"Error occured while reading from: {self.CollectionName}, {e}")

        return row


    def GetOneDocumentWithId(self, Qry = None):
        return self.GetOneDocument(1, Qry)


