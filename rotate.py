import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import modules.generator as GEN
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
from collections import OrderedDict
import pdb
if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num
    generator = getattr(GEN, opt.generator)(**config['model_params']['generator_params'],**config['model_params']['common_params'],**{'mbunit':opt.mbunit,'mb_spatial':opt.mb_spatial,'mb_channel':opt.mb_channel})
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


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    sources = []
    drivings = []
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        kp_source = kp_detector(source)
        if not cpu:
            kp_driving_initial = kp_detector(driving[:, :, 0].cuda())
        else:
            kp_driving_initial = kp_detector(driving[:, :, 0])
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return sources, drivings, predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num
















def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation_frontal(kp_source, yaw=0, pitch=0, roll=0):
    kp = kp_source

    yaw = torch.tensor([yaw]).float().cuda()
    pitch = torch.tensor([pitch]).float().cuda()
    roll = torch.tensor([roll]).float().cuda()
    
    rot_mat = get_rotation_matrix(yaw, pitch, roll)

    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    jacobian_transformed

    jacobian = kp_source['jacobian']
    jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)

    return {'value': kp_rotated, 'jacobian': jacobian_transformed}


def transform_front(source_image,generator, kp_detector, he_estimator, yaw, pitch, roll):
    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.cuda()
        kp_source = kp_detector(source)

        kp_source_frontal = keypoint_transformation_frontal(kp_source, yaw=yaw, pitch=pitch, roll=roll)
        out = generator(source, kp_source=kp_source, kp_driving=kp_source_frontal)

        frontal = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
    return frontal


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

def get_front_transform3Dangles(detection_result):
    cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(detection_result.facial_transformation_matrixes[0][:3,:])
    yaw=eulerAngles[0,0]
    pitch=eulerAngles[1,0]
    roll=eulerAngles[2,0]

    return yaw, pitch, roll

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


def frontalize_and_mirror(source_image, with_3d_rotate=True):

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
    yaw,pitch,roll= get_front_transform3Dangles(rotated_landmark_detection_result)
    nose_x, nose_y = rotated_landmark_detection_result.face_landmarks[0][4].x, rotated_landmark_detection_result.face_landmarks[0][4].y
    top_x, top_y = rotated_landmark_detection_result.face_landmarks[0][10].x, rotated_landmark_detection_result.face_landmarks[0][10].y

    if with_3d_rotate and np.abs(nose_x - top_x) > 0.008:
        # 3D face rotation neural talking head
        generator, kp_detector, he_estimator = load_checkpoints(config_path="vox-256-spade.yaml", checkpoint_path="00000189-checkpoint.pth.tar", gen="spade", cpu=False)

        # with open(opt.config) as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)
        # estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
        # print(f'estimate jacobian: {estimate_jacobian}')


        if nose_x - top_x < 0:
            yaw = -1 * yaw
        frontal = transform_front(image_cropped, generator, kp_detector, he_estimator, yaw,0,0)
        # frontal_detector_result = get_mediapipe_face_landmarks((frontal * 255).astype(np.uint8).copy(),face_landmark_detector)
        # yaw,pitch,roll= get_front_transform3Dangles(frontal_detector_result)
        # print(yaw, pitch, roll)
        # frontal = transform_front(frontal, generator, kp_detector, he_estimator, 0,pitch,roll)

    else:
        frontal = image_cropped
    # Convert back to uint8 for interface consistency
    frontal *= 255.0
    frontal.round(out=frontal)
    frontal.clip(0, 255, out=frontal)
    frontal = frontal.astype(np.uint8)

    cv2.imwrite("debug_rot.png", cv2.cvtColor(frontal, cv2.COLOR_RGB2BGR))
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






















if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--kp_num", type=int, required=True)
    parser.add_argument("--mb_channel",type=int, default=512, help='depth mode')
    parser.add_argument("--mb_spatial",type=int, default=32, help='depth mode')
    parser.add_argument("--mbunit",type=str, default='', help='depth mode')
    parser.add_argument("--memsize",type=int, default=1, help='depth mode')

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()
    
    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        sources_forward, drivings_forward, predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        sources_backward, drivings_backward, predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
        sources = sources_backward[::-1] + sources_forward[1:]
        drivings = drivings_backward[::-1] + drivings_forward[1:]
    else:
        sources, drivings, predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [np.concatenate((img_as_ubyte(s),img_as_ubyte(d),img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)

