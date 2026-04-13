import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

# from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

from crop_webcam import FFHQFaceCropper, LatentPoseFaceCropper

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['model_params']['common_params']['num_kp'] = 15
    generator = getattr(GEN, "Unet_Generator_keypoint_aware")(**config['model_params']['generator_params'],**config['model_params']['common_params'],**{'mbunit':"ExpendMemoryUnit",'mb_spatial':32,'mb_channel':512})
    if not cpu:
        generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path,map_location="cuda:0")
    
    ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['generator'].items())
    generator.load_state_dict(ckp_generator)
    ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in checkpoint['kp_detector'].items())
    kp_detector.load_state_dict(ckp_kp_detector)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector




def get_mediapipe_face_landmarks(image, detector=None):
    if detector is None:
        detector = init_mediapipe_facedetector()
        
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(mp_image)
    
    return detection_result

def get_facenorm_affine_angles(landmark_detection_result, image):
    nose_x, nose_y = landmark_detection_result.face_landmarks[0][4].x, landmark_detection_result.face_landmarks[0][4].y
    top_x, top_y = landmark_detection_result.face_landmarks[0][10].x, landmark_detection_result.face_landmarks[0][10].y
    bottom_x, bottom_y = landmark_detection_result.face_landmarks[0][152].x, landmark_detection_result.face_landmarks[0][152].y

    h, w = image.shape[:2] 
    nose_coord_x = w * nose_x
    nose_coord_y = h * nose_y
    top_coord_x = w * top_x
    top_coord_y = h * top_y
    bottom_coord_x = w * bottom_x
    bottom_coord_y = h * bottom_y
    #print("Top coord: ", top_coord_x, top_coord_y)
    #print("Bottom coord: ", bottom_coord_x, bottom_coord_y)

    ## Rotate the image
    multip = -1
    P1 = np.array([top_coord_x, top_coord_y])
    P2 = np.array([bottom_coord_x, bottom_coord_y])

    # checks orientation of p vector & selects appropriate y_axis_vector
    if (P2[1] - P1[1]) < 0:
        y_axis_vector = np.array([0, -1])
    else:
        y_axis_vector = np.array([0, 1])


    if (P2[0] - P1[0]) < 0 and (P2[1] - P1[1]) :
        y_axis_vector = np.array([0, 1])
        multip = 1

    p_unit_vector = (P2 - P1) / np.linalg.norm(P2-P1)
    
    angle_p_y = np.arccos(np.dot(p_unit_vector, y_axis_vector)) * 180 /math.pi
    rot_center = (nose_coord_x, nose_coord_y)
    
    return rot_center, angle_p_y, multip

def init_mediapipe_facedetector():
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    return detector

def rotate_image(image, angle, center = None, scale = 1.0):
    ''' This method rotates a given image by the provided angle.
    '''
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def crop_with_padding(image, t, l, b, r, segmentation=False):
    """
        image:
            numpy, np.uint8, (H x W x 3) or (H x W)
        t, l, b, r:
            int
        segmentation:
            bool
            Affects padding.

        return:
            numpy, (b-t) x (r-l) x 3
    """
    t_clamp, b_clamp = max(0, t), min(b, image.shape[0])
    l_clamp, r_clamp = max(0, l), min(r, image.shape[1])
    image = image[t_clamp:b_clamp, l_clamp:r_clamp]

    # If the bounding box went outside of the image, restore those areas by padding
    padding = [t_clamp - t, b - b_clamp, l_clamp - l, r - r_clamp]
    if sum(padding) == 0: # = if the bbox fully fit into image
        return image

    image = cv2.copyMakeBorder(image, *padding, cv2.BORDER_CONSTANT)
    assert image.shape[:2] == (b - t, r - l)

    # We will blur those padded areas
    h, w = image.shape[:2]
    y, x = map(np.float32, np.ogrid[:h, :w]) # meshgrids

    mask_l = np.full_like(x, np.inf) if padding[2] == 0 else (x / padding[2])
    mask_t = np.full_like(y, np.inf) if padding[0] == 0 else (y / padding[0])
    mask_r = np.full_like(x, np.inf) if padding[3] == 0 else ((w-1-x) / padding[3])
    mask_b = np.full_like(y, np.inf) if padding[1] == 0 else ((h-1-y) / padding[1])

    # The farther from the original image border, the more blur will be applied
    mask = np.maximum(
        1.0 - np.minimum(mask_l, mask_r),
        1.0 - np.minimum(mask_t, mask_b))

    # Do blur
    sigma = h * 0.016
    kernel_size = 0
    image_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Now we'd like to do alpha blending math, so convert to float32
    def to_float32(x):
        x = x.astype(np.float32)
        x /= 255.0
        return x
    image = to_float32(image)
    image_blurred = to_float32(image_blurred)

    # Support 2-dimensional images (e.g. segmentation maps)
    if image.ndim < 3:
        image.shape += (1,)
        image_blurred.shape += (1,)
    mask.shape += (1,)

    # Replace padded areas with their blurred versions, and apply
    # some quickly fading blur to the inner part of the image
    image += (image_blurred - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)

    fade_color = np.median(image, axis=(0,1))
    image += (fade_color - image) * np.clip(mask, 0.0, 1.0) 

    # Convert back to uint8 for interface consistency
    image *= 255.0
    image.round(out=image)
    image.clip(0, 255, out=image)
    image = image.astype(np.uint8)

    return image

def get_face_crop_box(detection_result, image, SCALE = 1.7):
    landmarks_2d = np.array([[x.x, x.y] for x in detection_result.face_landmarks[0]]) * np.array([[image.shape[1], image.shape[0]]])
    r,b = np.max(landmarks_2d, axis=0)
    l,t = np.min(landmarks_2d, axis=0)

    center_x, center_y = (l + r) * 0.5, (t + b) * 0.5
    height, width = b - t, r - l
    new_box_size = max(height, width)
    l = center_x - new_box_size / 2 * SCALE
    r = center_x + new_box_size / 2 * SCALE
    t = center_y - new_box_size / 2 * SCALE
    b = center_y + new_box_size / 2 * SCALE

    # Make floats integers
    l, t = map(math.floor, (l, t))
    r, b = map(math.ceil, (r, b))

    # After rounding, make *exactly* square again
    b += (r - l) - (b - t)
    assert b - t == r - l

    # Make `r` and `b` C-style (=exclusive) indices
    r += 1
    b += 1

    shift = -40

    t += shift
    b += shift
    
    return t, l, b, r


def get_nose_face_crop_box(detection_result, image, SCALE = 1.7, box_radius=None):
    nose_x, nose_y = detection_result.face_landmarks[0][4].x * image.shape[1], detection_result.face_landmarks[0][4].y *  image.shape[0]
    center_x, center_y = nose_x, nose_y
    if box_radius is not None:
        l = center_x - box_radius
        r = center_x + box_radius
        t = center_y - box_radius
        b = center_y + box_radius

    else:
        landmarks_2d = np.array([[x.x, x.y] for x in detection_result.face_landmarks[0]]) * np.array([[image.shape[1], image.shape[0]]])
        r,b = np.max(landmarks_2d, axis=0)
        l,t = np.min(landmarks_2d, axis=0)

        
        height, width = b - t, r - l
        new_box_size = max(height, width)
        
        box_radius = new_box_size / 2 * SCALE
        l = center_x - box_radius
        r = center_x + box_radius
        t = center_y - box_radius
        b = center_y + box_radius

    # Make floats integers
    l, t = map(math.floor, (l, t))
    r, b = map(math.ceil, (r, b))

    # After rounding, make *exactly* square again
    b += (r - l) - (b - t)
    assert b - t == r - l

    # Make `r` and `b` C-style (=exclusive) indices
    r += 1
    b += 1

    shift = -40

    t += shift
    b += shift
    
    return t, l, b, r, box_radius



def crop_face(image, box_radius=None):
    face_landmark_detector = init_mediapipe_facedetector()
    landmark_detection_result = get_mediapipe_face_landmarks(image,face_landmark_detector)
    # t, l, b, r = get_face_crop_box(landmark_detection_result, image)
    t, l, b, r, box_radius = get_nose_face_crop_box(landmark_detection_result, image, box_radius=box_radius)
    image_cropped = crop_with_padding(image, t, l, b, r)

    image_cropped = resize(image_cropped, (256, 256))[..., :3]

    return image_cropped, box_radius


def crop_video(input_video, destination):
    video_capture = cv2.VideoCapture(input_video)

    retval = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_length = None if retval < 0 else retval

    # Initialize image writer
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(fps, vid_length)
    if fps <= 0:
        fps = None

    i = 0
    video_frames = []
    crop_padding = None
    while True:
        success, input_image = video_capture.read()
        if not success:
            break
           
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        if fps is not None and fps > 20:
            if  i % (int(fps) // 20) != 0:
                i += 1 
                continue

        if max(input_image.shape) > 1152:
            resize_ratio = 1152 / max(input_image.shape)
            input_image = cv2.resize(input_image, dsize=None, fx=resize_ratio, fy=resize_ratio)

        image_cropped, crop_padding = crop_face(input_image, box_radius=crop_padding)
        video_frames.append(cv2.flip(image_cropped, 1))
        
        i += 1 

    if fps > 20:
        fps = 20
    imageio.mimsave(destination, [img_as_ubyte(frame) for frame in video_frames], fps=fps)


def frontalize(source_image):

    face_landmark_detector = init_mediapipe_facedetector()
    
    # affine face rotation
    landmark_detection_result = get_mediapipe_face_landmarks(source_image,face_landmark_detector)
    rot_center, angle_p_y, multip = get_facenorm_affine_angles(landmark_detection_result, source_image)
    image_rotated = rotate_image(source_image, multip * angle_p_y, center = rot_center)

    
    # cropped = cv2.resize(cropped, (256,256),
    #             interpolation=cv2.INTER_AREA)

    # 3D face rotation neural talking head
    rotated_landmark_detection_result = get_mediapipe_face_landmarks(image_rotated,face_landmark_detector)
    yaw,pitch,roll= get_front_transform3Dangles(rotated_landmark_detection_result)
    t, l, b, r = get_face_crop_box(rotated_landmark_detection_result, image_rotated)
    image_cropped = crop_with_padding(image_rotated, t, l, b, r)

    image_cropped = resize(image_cropped, (256, 256))[..., :3]
    generator, kp_detector, he_estimator = load_checkpoints(config_path="vox-256-spade.yaml", checkpoint_path="00000189-checkpoint.pth.tar", gen="spade", cpu=False)

    # with open(opt.config) as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    # print(f'estimate jacobian: {estimate_jacobian}')

    frontal = transform_front(image_cropped, generator, kp_detector, he_estimator, yaw,pitch,roll)

    return frontal

def mirror_face(image, detector=None):
    if detector is None:
        detector = init_mediapipe_facedetector()
    detection_result = get_mediapipe_face_landmarks(image,detector)
    h,w = image.shape[:2]
    x_crop = math.floor(w * detection_result.face_landmarks[0][10].x)

    fark = abs(math.floor(w * (detection_result.face_landmarks[0][152].x - detection_result.face_landmarks[0][10].x)))
    crop_img_l = image[0:h, 0:x_crop-(fark)]
    crop_img_r = image[0:h, x_crop-(fark):w]

    #plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))

    image_flip_l = cv2.flip(crop_img_l, 1)
    image_flip_r = cv2.flip(crop_img_r, 1)

    w - crop_img_l.shape[1]

    mirrored_image_l = cv2.hconcat([crop_img_l, image_flip_l[:,:w - crop_img_l.shape[1]]])
    

    mirrored_image_r = cv2.hconcat([image_flip_r[:,-(w - crop_img_r.shape[1]):], crop_img_r])


    # mirrored_image_l = cv2.hconcat([crop_img_l, image_flip_l])
    

    # mirrored_image_r = cv2.hconcat([image_flip_r, crop_img_r])
    
    
    return mirrored_image_l, mirrored_image_r, detection_result


def frontalize_and_mirror(source_image, with_3d_rotate=False):

    face_landmark_detector = init_mediapipe_facedetector()
    
    # affine face rotation
    landmark_detection_result = get_mediapipe_face_landmarks(source_image, face_landmark_detector)
    rot_center, angle_p_y, multip = get_facenorm_affine_angles(landmark_detection_result, source_image)
    image_rotated = rotate_image(source_image, multip * angle_p_y, center = rot_center)

    
    # cropped = cv2.resize(cropped, (256,256),
    #             interpolation=cv2.INTER_AREA)

    rotated_landmark_detection_result = get_mediapipe_face_landmarks(image_rotated,face_landmark_detector)
    t, l, b, r = get_face_crop_box(rotated_landmark_detection_result, image_rotated)
    image_cropped = crop_with_padding(image_rotated, t, l, b, r)
    image_cropped = resize(image_cropped, (256, 256))[..., :3]
    # import pdb; pdb.set_trace()
    # yaw,pitch,roll= get_front_transform3Dangles(rotated_landmark_detection_result)
    nose_x, nose_y = rotated_landmark_detection_result.face_landmarks[0][4].x, rotated_landmark_detection_result.face_landmarks[0][4].y
    top_x, top_y = rotated_landmark_detection_result.face_landmarks[0][10].x, rotated_landmark_detection_result.face_landmarks[0][10].y

    frontal = image_cropped
    # Convert back to uint8 for interface consistency
    frontal *= 255.0
    frontal.round(out=frontal)
    frontal.clip(0, 255, out=frontal)
    frontal = frontal.astype(np.uint8)

    # cv2.imwrite("debug_rot.png", cv2.cvtColor(frontal, cv2.COLOR_RGB2BGR))
    # import pdb; pdb.set_trace()
    mirrored_image_l, mirrored_image_r, detection_result = mirror_face(frontal.copy(), face_landmark_detector)

    return frontal, mirrored_image_l, mirrored_image_r, detection_result


def crop_image(source_image, face_landmark_detector):
    # 3D face rotation neural talking head
    rotated_landmark_detection_result = get_mediapipe_face_landmarks(source_image,face_landmark_detector)
    t, l, b, r = get_face_crop_box(rotated_landmark_detection_result, source_image)
    image_cropped = crop_with_padding(source_image, t, l, b, r)
    image_cropped = cv2.resize(image_cropped, (256, 256))

    return image_cropped

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cropper = LatentPoseFaceCropper((256, 256))
    while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if max(image.shape) > 1152:
            resize_ratio = 1152 / max(image.shape)
            image = cv2.resize(image, dsize=None, fx=resize_ratio, fy=resize_ratio)

        
        image_cropped, extra_data = cropper.crop_image(image)
        
        cv2.imshow('MediaPipe Hands', image_cropped)
        if cv2.waitKey(5) & 0xFF == 27:
          break
