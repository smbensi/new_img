import numpy as np
import cv2

from img_xtend.pose_estimation import keypoints
from img_xtend.pipelines.face_recognition.configuration import config as cfg

# def get_face_crop(img,x,y,h,w):
#     face_crop = (x,y, x+w, y+h)
#     face = jetson_utils.cudaAllocMapped(width=w, height=h, format=img.format)
#     jetson_utils.cudaCrop(img, face, face_crop)
    
#     return jetson_utils.cudaToNumpy(face)

# def face_in_frame(img,pose)-> bool:
#     '''
#         check if there is a face using the pose estimation result
#     '''
#     # check if there is nose and 2 eyes in the pose
#     nose_idx = pose.FindKeypoint(0)
#     left_eye_idx = pose.FindKeypoint(1)
#     right_eye_idx = pose.FindKeypoint(2)

#     if nose_idx < 0 or left_eye_idx<0 or right_eye_idx<0:
#         return False

#     # if there is a face but not in the ROI that we defined we'll return False
#     if pose.Keypoints[0].x > img.shape[1]*(1-cfg.MARGIN_IN_FRAME_TO_OMIT) \
#       or pose.Keypoints[0].x < img.shape[1]*cfg.MARGIN_IN_FRAME_TO_OMIT:
#         return False
    
#     return True


def get_face_from_pose(pose_results,img):
    keypoints_result = pose_results.keypoints if  hasattr(pose_results,'keypoints') else None
    if keypoints_result is None:
        raise IndexError("No keypoints in the results")
    face_all_data = {}
    for keypoint in keypoints_result:
        nose = keypoints.check_nose(keypoint)
        eyes = keypoints.check_eyes(keypoint)
        if nose is None or eyes is None:
            continue
        x, y, w, h = find_face(nose, eyes[0], eyes[1], img.shape )
        if h <= 0 or w <= 0:
                continue
        face = img[y:y+h,x:x+w]
        face_all_data["face"] = face        
        face_all_data["posn"] = [x+w//2,y+h//2,w,h]
    return face_all_data

def find_face(nose,left_eye,right_eye,img_shape):
    """The function find a bounding box of the face
    given two eyes' and  nose's coordinates

    Args:
        nose ([type]): [description]
        left_eye ([type]): [description]
        right_eye ([type]): [description]

    Returns:
        [type]: [description]
    """
    left_eye = left_eye.numpy()
    right_eye = right_eye.numpy()
    nose = nose.numpy()
    
    dist_eye_x  = left_eye[0] - right_eye[0]
    print(f"{dist_eye_x=}")
    dist_eye_y  = left_eye[1] - right_eye[1]
    print(f"{dist_eye_y=}")
    
    if dist_eye_x == 0:
        dist_eye_x = 20

    slope_eye = dist_eye_y/dist_eye_x
    print(f"{slope_eye=}")
    
    b_left = left_eye[1] - slope_eye*left_eye[0]
    b_right = left_eye[1] - slope_eye*right_eye[0]

    hypo = np.sqrt(dist_eye_x**2 + dist_eye_y**2)

    proportion = 2
    x_left_new = left_eye[0] - proportion*dist_eye_x
    y_left_new = left_eye[1] - proportion*dist_eye_y

    x_right_new = right_eye[0] + proportion*dist_eye_x
    y_right_new = right_eye[1] + proportion*dist_eye_y

    x_final = np.max((0,np.round(x_left_new).astype(int)))

    width = np.min((img_shape[1]-x_final-1, np.round(x_right_new - x_left_new).astype(int)))

    x_m = right_eye[0] + 0.5*dist_eye_x
    y_m = right_eye[1] + 0.5*dist_eye_y

    dist_nose_x = nose[0] - x_m
    dist_nose_y = nose[1] - y_m

    proportion2 = 4
    x_high_new = nose[0] - proportion2*dist_nose_x
    y_high_new = nose[1] - proportion2*dist_nose_y

    x_down_new = x_m + (proportion2)*dist_nose_x
    y_down_new = y_m + proportion2*dist_nose_y

    y_final = np.max((0,np.round(y_high_new).astype(int)))
    height = np.min((img_shape[0]-y_final-1, np.round(y_down_new - y_high_new).astype(int)))

    return x_final , y_final , height , width

def preprocess_for_recognition(face):
    image_size = 160
    face_resized = cv2.resize(face,(image_size,image_size), 
                            interpolation=cv2.INTER_AREA).copy()
    face_resized = (face_resized-127.5)/128
    face_resized = np.transpose(face_resized, [2,0,1]).astype('float32')
    return face_resized
