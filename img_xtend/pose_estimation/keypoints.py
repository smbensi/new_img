'''
In the default YOLOv8 pose model, there are 17 keypoints, 
each representing a different part of the human body. 
Here is the mapping of each index to its respective body joint:

0: Nose 
1: Left Eye 2: Right Eye 
3: Left Ear 4: Right Ear 
5: Left Shoulder 6: Right Shoulder 
7: Left Elbow 8: Right Elbow
9: Left Wrist 10: Right Wrist 
11: Left Hip 12: Right Hip 
13: Left Knee 14: Right Knee 
15: Left Ankle 16: Right Ankle

Example of keypoint output from ultralytics pose estimation model

ultralytics.engine.results.Keypoints object with attributes:

conf: tensor([[0.9945, 0.9954, 0.9888, 0.9686, 0.8300, 0.9808, 0.9360, 0.3005, 0.1100, 0.3800, 0.1831, 0.0309, 0.0181, 0.0044, 0.0032, 0.0018, 0.0016]])
data: tensor([[[2.7242e+02, 2.5695e+02, 9.9449e-01],
         [3.0583e+02, 2.2246e+02, 9.9540e-01],
         [2.4768e+02, 2.2208e+02, 9.8877e-01],
         [3.5742e+02, 2.2924e+02, 9.6858e-01],
         [2.1532e+02, 2.2666e+02, 8.3003e-01],
         [4.5355e+02, 3.7768e+02, 9.8075e-01],
         [1.0484e+02, 3.6682e+02, 9.3603e-01],
         [0.0000e+00, 0.0000e+00, 3.0048e-01],
         [0.0000e+00, 0.0000e+00, 1.0998e-01],
         [0.0000e+00, 0.0000e+00, 3.8000e-01],
         [0.0000e+00, 0.0000e+00, 1.8306e-01],
         [0.0000e+00, 0.0000e+00, 3.0854e-02],
         [0.0000e+00, 0.0000e+00, 1.8136e-02],
         [0.0000e+00, 0.0000e+00, 4.3763e-03],
         [0.0000e+00, 0.0000e+00, 3.2417e-03],
         [0.0000e+00, 0.0000e+00, 1.7613e-03],
         [0.0000e+00, 0.0000e+00, 1.5504e-03]]])
has_visible: True
orig_shape: (480, 640)
shape: torch.Size([1, 17, 3])
xy: tensor([[[272.4166, 256.9484],
         [305.8342, 222.4635],
         [247.6758, 222.0833],
         [357.4155, 229.2363],
         [215.3175, 226.6556],
         [453.5510, 377.6779],
         [104.8393, 366.8242],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000],
         [  0.0000,   0.0000]]])
xyn: tensor([[[0.4257, 0.5353],
         [0.4779, 0.4635],
         [0.3870, 0.4627],
         [0.5585, 0.4776],
         [0.3364, 0.4722],
         [0.7087, 0.7868],
         [0.1638, 0.7642],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000],
         [0.0000, 0.0000]]])

'''
import torch

BODY_JOINT={0: "Nose", 
1: "Left Eye",          2: "Right Eye", 
3: "Left Ear",          4: "Right Ear", 
5: "Left Shoulder",     6: "Right Shoulder", 
7: "Left Elbow",        8: "Right Elbow",
9: "Left Wrist",        10: "Right Wrist", 
11: "Left Hip",         12: "Right Hip", 
13: "Left Knee",        14: "Right Knee", 
15: "Left Ankle",       16: "Right Ankle",
}

def check_nose(keypoint):
    """return the xy coordinate if the nose is in the image o.w None"""
    if keypoint.conf[0,0] > 0.5:
        return keypoint.xy[0,0,:]
    else:
        return None

def check_eyes(keypoint):
    # 1: Left Eye 2: Right Eye 
    if keypoint.conf[0,1] > 0.5 and keypoint.conf[0,2] > 0.5:
        return torch.cat([keypoint.xy[0,1,:].unsqueeze(0),keypoint.xy[0,2,:].unsqueeze(0)], dim=0)
    else:
        return None

def check_shoulders(keypoint):
    # 5: Left Shoulder 6: Right Shoulder 
    if keypoint.conf[0,5] > 0.7 and keypoint.conf[0,6] > 0.7:
        return torch.cat([keypoint.xy[0,5,:].unsqueeze(0),keypoint.xy[0,6,:].unsqueeze(0)], dim=0)
    else:
        return None
    
def check_ears(keypoint):
    if keypoint.conf[0,3] > 0.5 and keypoint.conf[0,4] > 0.5:
        return torch.cat([keypoint.xy[0,3,:].unsqueeze(0),keypoint.xy[0,4,:].unsqueeze(0)], dim=0)
    else:
        return None
    
def check_person_state(keypoint):
    """Check if the person is facing / showing his back / showing profile to the camera
    based on the pose estimation points"""
    if keypoint == []:
        return None
    if check_eyes(keypoint) is not None:
        return PersonState.Face
    elif check_eyes(keypoint) is None and check_shoulders(keypoint) is not None:
        return PersonState.Back
    else:
        return PersonState.Profile


class PersonState:
    
    Face = 1
    Back = 2
    Profile = 3