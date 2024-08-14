import abc

class BaseFollow(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError("Subclasses must implement update()")


class FollowTracker:
    """
    This class is responsible to follow one person through the frames and also responsible to reidentify him if it disapear and reappear
    We are not using Kalman Filter to estimatee where the person moved it we have not identify it in the present frames
    
    Args:
    
    
    Methods:
        __init__(): check that the re-id model is loaded and init the Tracker object
        start_follow():
        update(): based on the update of the tracks and the id followed update
        save_img(): for debugging save img to check teh results
    
    """
    def __init__(
        self,
        max_dist=0.2,
        max_iou_dist = 0.7,
        max_age=30,
        n_init = 1,
        nn_budget=100,
        mc_lambda = 0.995,
        ema_alpha=0.9
        ):
        
        # parameters for integration
        self.track_id_followed = None
        self.following = False
        
        # parameters for track identification
        self.height = None
        self.width = None
        self.hits = 0
        self.miss = 0
        self.last_time_update = None
        self.previous_bbox = None
        self.same_bbox = None
        
        
        # parameters for the Tracker object
        self.max_dist = max_dist
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.mc_lambda = mc_lambda
        self.ema_alpha = ema_alpha
        
        # function to re/initialize the tracker
        self.initialize_tracker()
        
    
    def initialize_tracker(self):
        self.tracker = Tracker(
            metric = NearestNeighborDistanceMetric("cosine", self.max_dist, self.nn_budget),
            max_iou_dist=self.max_iou_dist,
            max_age=self.max_age,
            n_init=self.n_init,
            mc_lambda=self.mc_lambda,
            ema_alpha=self.ema_alpha
        )
        
    def set_followed_track(self, bboxes):
        pass
    
    def update(self, bboxes):
        if not self.following:
            self.set_followed_track(bboxes)
        else:
            pass