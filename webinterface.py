import gradio as gr
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import frontalize
import subprocess
import importlib  
import driver
crop_video = importlib.import_module("crop-video")
NO_CACHING = True
def get_half_face_masks(detection_result):
    landmarks = (np.array([[x.x, x.y] for x in detection_result.face_landmarks[0]])*256).astype(int)
    
    # landmark indices
    subset_landmarks_right = landmarks[[9, 18, 336, 296,334,293,300,197,195,5,4,1,17,314,405,321,375,432,427,340,280]]
    subset_landmarks_left = landmarks[[9, 18, 107, 66,105,63,70,139,50,207,57,43,106,182]]
    
    cvxhull_right = cv2.convexHull(subset_landmarks_right)
    cvxhull_left = cv2.convexHull(subset_landmarks_left)

    contour_mask_right = cv2.drawContours(np.zeros((256,256)), [cvxhull_right], -1, 255, thickness=cv2.FILLED)
    contour_mask_left = cv2.drawContours(np.zeros((256,256)), [cvxhull_left], -1, 255, thickness=cv2.FILLED)
    contour_mask_right = (contour_mask_right > 128).astype(float)
    contour_mask_left = (contour_mask_left > 128).astype(float)
    
    combined_mask_left = ((cv2.flip(contour_mask_right, 1) + contour_mask_left) >= 1).astype(int)
    combined_mask_right = ((cv2.flip(contour_mask_left, 1) + contour_mask_right) >= 1).astype(int)
    
    return combined_mask_left, combined_mask_right, contour_mask_left, contour_mask_right

def blend_side_face(front_image, inpainted_image, mask, direction="right", poisson=True, dilate=True, dilate_kernel=11):
    
    flipped_face, mask_flipped = flip_mask_and_cropped_face(front_image, mask, direction)
    
    
    if dilate:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8) 
        mask_padded = cv2.dilate((mask_flipped * 255).astype("uint8"), kernel, iterations=1) / 255.
    else:
        mask_padded = mask_flipped
        
        
    cropped_side_face = (flipped_face * mask_padded[:,:,None]).astype("uint8")
    coords = np.stack(np.where(mask_padded  > 0.9),axis=1)
    ymin, xmin = np.min(coords, axis = 0)
    ymax, xmax = np.max(coords, axis = 0)
    center = [(xmax + xmin)//2,(ymax + ymin)//2 ]
    mask_poisson = ((mask_padded > 0.9).astype(float) *255).astype("uint8")
    
    
    if poisson:
        blended1 = cv2.seamlessClone(cropped_side_face, inpainted_image, mask_poisson, center, cv2.NORMAL_CLONE)
    else:
        blended1 = flipped_face

    sigma = 3
    kernel_size = 11
    contour_mask_blurred = cv2.GaussianBlur((mask_padded * 255).astype("uint8"), (kernel_size, kernel_size), sigma) / 255
    blended2 = inpainted_image *(1 - contour_mask_blurred[:,:,None])+ blended1[:,:]*contour_mask_blurred[:,:,None]
    
    return blended2.astype("uint8"), cropped_side_face

def flip_mask_and_cropped_face(face_image, mask, mask_direction="right"):
    H,W = mask.shape
    orig_left_pt = np.min(np.where(np.sum(mask, axis=0) > 0))
    orig_right_pt = np.max(np.where(np.sum(mask, axis=0) > 0))
    
    face_flipped = cv2.flip(face_image,1)
    mask_flipped = cv2.flip(mask,1)
    
    right_pt = np.max(np.where(np.sum(mask_flipped, axis=0) > 0))
    left_pt = np.min(np.where(np.sum(mask_flipped, axis=0) > 0))
    mask_width = right_pt - left_pt
    tmp_mask = np.zeros_like(mask)
    tmp_face = np.zeros_like(face_image)
    
    
    
    if mask_direction == "right":
        tmp_mask[:,orig_left_pt - mask_width:orig_left_pt] = mask_flipped[:,left_pt:right_pt]
        tmp_face[:,orig_left_pt - mask_width:orig_left_pt] = face_flipped[:,left_pt:right_pt]
        
        if orig_left_pt - mask_width > left_pt:
            st = orig_left_pt - mask_width - left_pt
            tmp_face[:,st:orig_left_pt - mask_width] = face_flipped[:,:left_pt]
        else:
            st = left_pt - (orig_left_pt - mask_width)
            tmp_face[:,:orig_left_pt - mask_width] = face_flipped[:,st:left_pt]

        if orig_left_pt > right_pt:
            end = orig_left_pt - right_pt
            tmp_face[:,orig_left_pt:] = face_flipped[:,right_pt:W - end]
        else:
            end = right_pt - orig_left_pt
            tmp_face[:,orig_left_pt:W - end] = face_flipped[:,right_pt:]
            
    else:
        tmp_mask[:,orig_right_pt + 1:orig_right_pt + mask_width + 1] = mask_flipped[:,left_pt:right_pt]
        tmp_face[:,orig_right_pt + 1:orig_right_pt + mask_width + 1] = face_flipped[:,left_pt:right_pt]
        
        if orig_right_pt + 1 > left_pt:
            st = orig_right_pt + 1 - left_pt
            tmp_face[:,st:orig_right_pt + 1] = face_flipped[:,:left_pt]
        else:
            st = left_pt - (orig_right_pt + 1)
            tmp_face[:,:orig_right_pt + 1] = face_flipped[:,st:left_pt]

        if orig_right_pt + mask_width + 1 > right_pt:
            end = orig_right_pt + mask_width + 1 - right_pt
            tmp_face[:,orig_right_pt + mask_width + 1:] = face_flipped[:,right_pt:W - end]
        else:
            end = right_pt - (orig_right_pt + mask_width + 1)
            tmp_face[:,orig_right_pt + mask_width + 1:W - end] = face_flipped[:,right_pt:]
        
    mask_flipped = tmp_mask
    face_flipped = tmp_face
    return face_flipped.astype("uint8"), mask_flipped

def process_identity(
    input_identity, identity=None
  ):
    
    identity = identity.strip()
    
    with torch.no_grad():
        if identity is None or identity == "":
            randint = np.random.randint(0,99999)
            identity = "identity_{:05d}".format(randint)
        iden_dst = "identity-processed/{}".format(identity)

        
        if not os.path.exists(iden_dst) or NO_CACHING:
            os.makedirs(iden_dst,exist_ok=True)
            
            image = input_identity
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            frontal_image, mirror_l, mirror_r, detection_result = frontalize.frontalize_and_mirror(image,with_3d_rotate=False)
            combined_mask_left, combined_mask_right, contour_mask_left, contour_mask_right = get_half_face_masks(detection_result)

            inpaint_left = cv2.inpaint(frontal_image, (contour_mask_left * 255).astype("uint8"), 3, cv2.INPAINT_TELEA)
            inpaint_right = cv2.inpaint(frontal_image, (contour_mask_right * 255).astype("uint8"), 3, cv2.INPAINT_TELEA)

            blended_left, cropped_side_face_left = blend_side_face(frontal_image, inpaint_right, contour_mask_left, direction="left", poisson=True,  dilate=False)
            blended_right,cropped_side_face_right = blend_side_face(frontal_image, inpaint_left, contour_mask_right, direction="right", poisson=True, dilate=False)
           
            frontal_path = "{}/{}-frontal.png".format(iden_dst,identity)
            blended_left_path = "{}/{}-blended_left.png".format(iden_dst,identity)
            blended_right_path = "{}/{}-blended_right.png".format(iden_dst,identity)
            mirror_l_path = "{}/{}-mirror_l.png".format(iden_dst,identity)
            mirror_r_path = "{}/{}-mirror_r.png".format(iden_dst,identity)
            cv2.imwrite(frontal_path, cv2.cvtColor(frontal_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(blended_left_path, cv2.cvtColor(blended_left, cv2.COLOR_RGB2BGR))
            cv2.imwrite(blended_right_path, cv2.cvtColor(blended_right, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mirror_l_path, cv2.cvtColor(mirror_l, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mirror_r_path, cv2.cvtColor(mirror_r, cv2.COLOR_RGB2BGR))
            outputs = [(frontal_path, "Front"),(blended_left_path, "Blend Left"), (blended_right_path, "Blend Right"),(mirror_l_path, "Mirror Left"),(mirror_r_path, "Mirror Right")]
    return outputs, identity




def mirror_face_image(input_image, identity, iden_dst = "mirrored_faces"):
    os.makedirs(iden_dst,exist_ok=True)


    image = input_image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    frontal_image, mirror_l, mirror_r, detection_result = frontalize.frontalize_and_mirror(image,with_3d_rotate=False)
    combined_mask_left, combined_mask_right, contour_mask_left, contour_mask_right = get_half_face_masks(detection_result)

    inpaint_left = cv2.inpaint(frontal_image, (contour_mask_left * 255).astype("uint8"), 3, cv2.INPAINT_TELEA)
    inpaint_right = cv2.inpaint(frontal_image, (contour_mask_right * 255).astype("uint8"), 3, cv2.INPAINT_TELEA)

    blended_left, cropped_side_face_left = blend_side_face(frontal_image, inpaint_right, contour_mask_left, direction="left", poisson=True,  dilate=False)
    blended_right,cropped_side_face_right = blend_side_face(frontal_image, inpaint_left, contour_mask_right, direction="right", poisson=True, dilate=False)


    frontal_path = "{}/{}-frontal.png".format(iden_dst,identity)
    blended_left_path = "{}/{}-blended_left.png".format(iden_dst,identity)
    blended_right_path = "{}/{}-blended_right.png".format(iden_dst,identity)
    mirror_l_path = "{}/{}-mirror_head_left.png".format(iden_dst,identity)
    mirror_r_path = "{}/{}-mirror_head_right.png".format(iden_dst,identity)
    cv2.imwrite(frontal_path, cv2.cvtColor(frontal_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(blended_left_path, cv2.cvtColor(blended_left, cv2.COLOR_RGB2BGR))
    cv2.imwrite(blended_right_path, cv2.cvtColor(blended_right, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mirror_l_path, cv2.cvtColor(mirror_l, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mirror_r_path, cv2.cvtColor(mirror_r, cv2.COLOR_RGB2BGR))


    blended_left, cropped_side_face_left = blend_side_face(frontal_image, inpaint_right, contour_mask_left, direction="left", poisson=False,  dilate=False)
    blended_right,cropped_side_face_right = blend_side_face(frontal_image, inpaint_left, contour_mask_right, direction="right", poisson=False, dilate=False)
   
    blended_left_path = "{}/{}-mirror_face_left.png".format(iden_dst,identity)
    blended_right_path = "{}/{}-mirror_face_right.png".format(iden_dst,identity)

    cv2.imwrite(frontal_path, cv2.cvtColor(frontal_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(blended_left_path, cv2.cvtColor(blended_left, cv2.COLOR_RGB2BGR))
    cv2.imwrite(blended_right_path, cv2.cvtColor(blended_right, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mirror_l_path, cv2.cvtColor(mirror_l, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mirror_r_path, cv2.cvtColor(mirror_r, cv2.COLOR_RGB2BGR))





def drive_video(input_driver, identity_image):
    iden = identity_image.split("-")[0]
    iden_dst = "identity-processed/{}".format(iden)
    image_path = "{}/{}".format(iden_dst,identity_image)
    output_path = "{}/{}-{}.mp4".format(iden_dst,iden, input_driver)
    driver_path = "driver_videos/{}.mp4".format(input_driver)
    driver.drive_function(image_path, driver_path, output_path)

    return output_path


def process_driver_video(
    video, driver_identity=None
  ):
    
    
    driver_identity = driver_identity.strip()
    
    if driver_identity is None or driver_identity == "" or NO_CACHING:
        randint = np.random.randint(0,99999)
        driver_identity = "driver_{:05d}".format(randint)
    with torch.no_grad():
        crop_dst = "driver_videos/{}.mp4".format(driver_identity)
        if not os.path.exists(crop_dst):
            os.makedirs("driver_videos", exist_ok=True)
            frontalize.crop_video(video, crop_dst)

    return crop_dst, driver_identity


def process_gallery(evt: gr.SelectData):
    # import pdb; pdb.set_trace()
    return evt.value["image"]["orig_name"]

if __name__ == '__main__':
    


    demo = gr.Blocks()

    with demo:
        with gr.Row():
            with gr.Column():   
                gr.Markdown("""
                        ## Step 1: Identity video / image
                    """)  
                
                with gr.Tabs():
                    with gr.TabItem(label="Capture an image"):
                        gr.Markdown("""
                            Record or upload an image to use as reference face identity.
                        """)  
                        image_iden = gr.Image(sources=["webcam", "upload"])
                        submit_image_iden = gr.Button(value="Process Identity Image")
                        identity_text = gr.Textbox(placeholder="Identity Name (Automatically generated)", interactive=False)

            with gr.Column():
                gr.Markdown("""
                        ## Step 2: Reference Motion video
                    """)
                with gr.Tabs():
                    with gr.TabItem(label="Record a video"):
                        gr.Markdown("""
                            Record or upload a video to use as motion reference.
                        """)  
                        recorded_video_driver = gr.Video(sources=["webcam", "upload"], format="mp4")
                        submit_recorded_video_driver = gr.Button(value="Process Driver Video")
                        ref_text = gr.Textbox(placeholder="Reference Name (Automatically generated)", interactive=False)

                    recorded_video_driver_output= gr.Video(label="Processed driver video", interactive=False)
            with gr.Column(): 
                gr.Markdown("""
                        ## Step 3: Puppeteering
                    """)
                gr.Markdown("""
                            Select the desired mirrored face for puppeteering.
                        """)  
                gallery = gr.Gallery(label="Generated images",allow_preview=False,selected_index=0)
                selected_image = gr.Textbox(placeholder="Selected Image (Automatically generated)", interactive=False)

                run_driver = gr.Button(value="Run puppeteering")                 
                processed_video = gr.Video(label="Out", interactive=False)
        

        submit_recorded_video_driver.click(fn=process_driver_video, inputs=[recorded_video_driver,ref_text], outputs=[recorded_video_driver_output, ref_text])
        # submit_uploaded_video_driver.click(fn=process_driver_video, inputs=[uploaded_video_driver,ref_text], outputs=[recorded_video_driver_output, ref_text])
        # submit_uploaded_video.click(fn=process_video, inputs=[recorded_video], outputs=[processed_video])
        

        # submit_recorded_video_driver.click(fn=process_driver_video, inputs=[recorded_video_driver, ref_text], outputs=[recorded_video_driver_output, ref_text])
        submit_image_iden.click(fn=process_identity, inputs=[image_iden, identity_text], outputs=[gallery, identity_text])


        run_driver.click(fn=drive_video, inputs=[ref_text, selected_image], outputs=[processed_video])

        gallery.select(fn=process_gallery, outputs=[selected_image])
    demo.queue().launch(server_name="0.0.0.0", server_port=7001, share=True)
    # demo.queue().launch(server_name="0.0.0.0", server_port=7001)
