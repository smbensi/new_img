from pymongo import MongoClient
import os
import json
# import xtendlog
from img_xtend.utils import LOGGER
# import MongoCollection
from bson.json_util import dumps


class MongoDB:
    ip = os.environ.get("mongo","172.17.0.1")
    mongo_string = (f"mongodb://{ip}:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false")
    # mongo_string = "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
    # self.logger.info(f"MogoDB IP: {mongo_string}")
    client = MongoClient(mongo_string)
    db = client.xtend_robotics

    def __init__(self) -> None:
        # self.logger = xtendlog.loggerclass.get()
        self.logger = LOGGER

    def get_db_instance(self):
        return self.db


    def DumpDB(self):
        self.logger.info("In DumpDB")
        DumpDB = {}
        for CollName in self.db.list_collection_names():
            Collection = self.db[CollName]
            DumpDB[CollName] = [doc for doc in Collection.find()]
            
        DumpDB = json.loads(dumps(DumpDB))
        return DumpDB


# old ip addresses for mongo. Just for safe keeping...
#        mongo_string = "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
#        mongo_string = "mongodb://172.17.0.1:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
#        mongo_string = "mongodb://host.docker.internal:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
