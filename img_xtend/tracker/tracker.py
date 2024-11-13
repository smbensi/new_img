from typing import List
import os 

import numpy as np

from img_xtend.utils import LOGGER, get_time, log_to_file
from img_xtend.detection.bbox import Bbox
from . import linear_assignment 
from img_xtend.tracker.matching import iou_matching
from img_xtend.tracker.matching.match_dataclass import MatchData
from img_xtend.tracker.check_embedding import add_new_detection
from img_xtend.tracker.track import Track
from img_xtend.tracker import tracker_settings
from img_xtend.pose_estimation import keypoints
from img_xtend.pipelines.follow_person import follow_settings
class Tracker:
    
    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        
        self.tracks: List[Track] = []
        self._next_id = 1
        # self.cmc = get_cmc_method('ecc')()
        
        self.last_number_of_detections = -1
        self.last_tracks_detected = -1
        self.last_matches_from = ""
        
    def predict(self):
        '''predict when using Kalman filter . NOT USED'''
        for track in self.tracks:
            track.predict()
    
    def increment_ages(self):
        
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def manual_match(self,bboxes, cost_matrix_log):
        matches = []
        logs_manual_match = ""
        
            
        if len(bboxes) == 1 and self.last_number_of_detections ==1:
            # for el in self.last_tracks_detected:
            for track in self.tracks:
                if track.id in self.last_tracks_detected and track.is_followed and track.miss == 0:
                    # LOGGER.debug(f"detections: {len(bboxes)}, previous detections: {self.last_number_of_detections} tracks detected: {self.last_tracks_detected} appearance matrix: {cost_matrix_log}")
                    
                    if len(cost_matrix_log)>0 and cost_matrix_log[0,0] < 1:
                        matches=[(self.last_tracks_detected.index(track.id),0)]
        logs_manual_match = f" detections: {len(bboxes)}, previous detections: {self.last_number_of_detections}"
        return matches,logs_manual_match
    
    def update(self, bboxes: List[Bbox]):
        '''Perform measurement update and track management
        compute the match between the bbox and the track and
        return matches, unmatched_tracks, unmatched_bbox, match_data
        for each match update the track
        for each unmatch track update the track to missed
        for each unmatch detection create a new track
        the logs data contains all the data about the matchs like IOU, SIMILARITY,...
        
        Parameters
        ----------
        bboxes: List[ultralytics.Bbox]
            A list of bboxes at the current time step
        '''
        # TODO : consider differently the followed_id        
        # iou_matrix = compute_iou_bboxes(bboxes)
        matches_manual, logs_manual_match = [], ""
        if tracker_settings.USE_MANUAL_MATCH:
            try:
                matches_manual, logs_manual_match = self.manual_match(bboxes, cost_matrix_log) 
            except Exception as e:
                LOGGER.debug(f"PROBLEM MANUAL MATCH: {e}")
                matches_manual =[]
                
        # Run matching cascade 
        matches, unmatched_tracks, unmatched_bbox, match_data = self._match(bboxes)
        cost_matrix_log = match_data.cost_matrix
        argmin_matrix = match_data.argmin_matrix
    
        matches_from = match_data.match_from 
        logs_matches_from = matches_from + " " + str(cost_matrix_log)
        
        if len(matches) == 0 and len(matches_manual) != 0:
            matches = matches_manual
            matches_from = "MANUAL" 
            logs_matches_from = matches_from + logs_manual_match
            for track_idx, bbox_idx in matches:
                unmatched_tracks.remove(track_idx) if track_idx in unmatched_tracks else None
                unmatched_bbox.remove(bbox_idx) if bbox_idx in unmatched_bbox else None     

        # LOGGER.debug(f"{matches=}")
        # matches contains tuples indicating association between tracks and bboxes in the frame
        for track_idx, bbox_idx in matches:
            add_embedding = False   
            if self.tracks[track_idx].is_followed:
                follow_settings.MATCHES_SOURCE[matches_from] += 1
                follow_settings.MATCHES_SOURCE["FRAMES"] += 1
                add_embedding = add_new_detection(matches_from, cost_matrix_log, bboxes, *match_data.index_in_cost_matrix)
                LOGGER.debug(f"{matches_from=} {cost_matrix_log=} {add_embedding=}")
                if matches_from != self.last_matches_from and matches_from in follow_settings.MATCHES_TO_PRINT:
                    # LOGGER.debug(f"match from: {logs_matches_from}")
                    self.last_matches_from = matches_from
                        
            try:
                similarity_val = cost_matrix_log[match_data.index_in_cost_matrix[0], match_data.index_in_cost_matrix[1]] if add_embedding else -100
                argmin_val = argmin_matrix[match_data.index_in_cost_matrix[0], match_data.index_in_cost_matrix[1]]
                self.tracks[track_idx].update(bboxes[bbox_idx],match_data,argmin_val, add_embedding=add_embedding , similarity_val=similarity_val, matches_from=matches_from)
            except IndexError as e:
                LOGGER.debug(f"ERROR IN UPDATE {e} \n {matches_from=} {cost_matrix_log=} \n {track_idx=},{bbox_idx=},{matches=} \n matches_appearance: {match_data.matches_appearance} \n matches_iou: {match_data.matches_iou},\n{argmin_matrix=} \n{self.tracks=}, \n{bboxes=}")
                
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for bbox_idx in unmatched_bbox:
            self._initiate_track(bboxes[bbox_idx])
            
        self.last_number_of_detections = len(bboxes)
        self.last_tracks_detected = [self.tracks[track_idx].id for track_idx,_ in matches]
        
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        
        # update distance metric
        # active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features                      # list of all the embeddings contained in tracks
        #     targets += [track.id for _ in track.features]   # list of list of repeated id for each features' element
        # # LOGGER.debug(f'{len(features)=} and {active_targets=} and {targets=}')
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets
        # )
        return self.tracks, match_data
    
    
    def _match_appearance(self, bboxes:Bbox, confirmed_tracks: Track):
        """_summary_

        Args:
            bboxes (Bbox): _description_
            confirmed_tracks (Track): _description_
        
        Returns:
            matches_appearance: Tuple[int, int]: the first element is the index of the
                track in self.tracks and the second element is the index of the bbox in bboxes
            
        """
        
        def gated_metric(tracks: List[Track], bboxes: List[Bbox], track_indices, bbox_indices):
            features = np.array([bboxes[i].emb for i in bbox_indices])
            # targets = np.array([tracks[i].id for i in track_indices])
            targets = np.array([tracks[i] for i in track_indices])
            # LOGGER.debug(f'TARGETS IN THE FRAME  ARE {targets}')
            cost_matrix = self.metric.distance(features, targets,tracks)
            # LOGGER.debug(f'GATED METRIC {cost_matrix=}')
            return cost_matrix
        
        matches_appearance, unmatched_tracks_appearance, unmatched_bboxes, logs_data_appearance = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            bboxes,
            confirmed_tracks,
        )
        
        return matches_appearance, unmatched_tracks_appearance, unmatched_bboxes, logs_data_appearance

    def _match_iou(self, bboxes, iou_tracks_candidates, unmatched_bboxes):
        """_summary_

        Args:
            bboxes (_type_): _description_
            tracks (_type_): _description_
        """
        try:
            matches_iou, unmatched_tracks_iou, unmatched_detections,logs_data_iou = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.tracks,
                bboxes,
                iou_tracks_candidates,
                unmatched_bboxes,
            )
        
            # if len(matches_b) > 0:
        except Exception as e:
            matches_iou, unmatched_tracks_iou, unmatched_detections = [], [], []
            LOGGER.debug(f"ERROR: {e}")

        return matches_iou, unmatched_tracks_iou, unmatched_detections, logs_data_iou
        
    def _match(self, bboxes):
        """
        Search for matches between existing tracks and bboxes detected in present frames according
        to different criteria: 
        MANUAL
        APPEARANCE: Compare between the embedding vectors of each track and the embedding vectors
                    of each bbox
        IOU: Compare between the position of each track and the position of each bbox if the track
            was in the NB_OF_MISS_IOU frames
        
        """
        # split track set into confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_appearance, unmatched_tracks_appearance, unmatched_bboxes, logs_data_appearance = self._match_appearance(bboxes, confirmed_tracks)
        
        appearance_cost = logs_data_appearance.get("cost_matrix_log",[])
        # Associate remaining tracks together with unconfirmed tracks using IOU
        iou_tracks_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_appearance if self.tracks[k].miss < tracker_settings.NB_OF_MISS_IOU
            ]
        
        matches_iou, unmatched_tracks_iou, unmatched_detections, logs_data_iou = self._match_iou(bboxes, iou_tracks_candidates, unmatched_bboxes)
        
        index_in_cost_matrix = (0,0)
        MATCHES = ""
        for track_idx,bbox_idx in matches_appearance:
            if self.tracks[track_idx].is_followed:
                MATCHES = "APPEARANCE"
                index_in_cost_matrix =(track_idx,bbox_idx)
        tracks_a = [track_idx for track_idx,_ in matches_appearance]
        for track_idx,bbox_idx in matches_iou :
            if track_idx not in tracks_a and self.tracks[track_idx].is_followed:
                MATCHES = "IOU"
                index_in_cost_matrix = (iou_tracks_candidates.index(track_idx), unmatched_bboxes.index(bbox_idx))
                logs_data_appearance = logs_data_iou
                sentence_to_log = f"\n{get_time()}:match: {track_idx, bbox_idx}   index: {index_in_cost_matrix} {logs_data_iou=} \n\
                    {iou_tracks_candidates=}, {bboxes=} \n \
                    track_idx{[i for i,_ in matches_iou]} and tracks={self.tracks[:3]} \n \
                    cost appearance in IOU match {appearance_cost} \n\n"
                log_to_file(follow_settings.IOU_LOGS, sentence_to_log)
        
        match_data = MatchData(match_from=MATCHES,
                               cost_matrix= logs_data_appearance.get("cost_matrix_log",[]),
                               argmin_matrix=logs_data_appearance.get("argmin_matrix", []),
                               index_in_cost_matrix=index_in_cost_matrix,
                               matches_appearance=matches_appearance,
                               matches_iou=matches_iou,
                               iou_cost=logs_data_iou.get("cost_matrix_log",[]),
                               appearance_cost=appearance_cost,
                               bboxes=bboxes,
                               tracks=self.tracks
                               )
                
        matches = matches_appearance + matches_iou
        tracks_b = [track_idx for track_idx,_ in matches_iou]
        unmatched_tracks_appearance = [el for el in unmatched_tracks_appearance if el not in tracks_b]
        unmatched_tracks = list(set(unmatched_tracks_appearance + unmatched_tracks_iou))
        
        return matches, unmatched_tracks, unmatched_detections, match_data
    
    def _initiate_track(self, bbox):
        self.tracks.append(
            Track(
                bbox,
                self._next_id,
                self.n_init,
                self.max_age,
                self.ema_alpha,
                track_followed=False
            )
            
        )
        self._next_id += 1
        
    def set_follow_track(self,index):
        for i, track in enumerate(self.tracks):
            if track.id == index:
                self.tracks[i].change_follow_status()
                LOGGER.debug(f'self.tracks[{i}]={self.tracks[i]}')