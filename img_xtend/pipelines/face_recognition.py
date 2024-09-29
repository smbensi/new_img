class FaceRecognition:
    def __init__(
        self,
        similarity_tresh,
        new_person_thresh,
        ):
        
        self.similarity_thresh = similarity_tresh
        self.new_person_thresh = new_person_thresh
        
        self.recognized_faces = []
        self.unrecognized_faces = []
        
    
    def update(self, img):
        pass