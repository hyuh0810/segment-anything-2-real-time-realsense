import torch
from sam2.build_sam import build_sam2_camera_predictor

import cv2
import numpy as np
import pyrealsense2 as rs

checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

# set realsense
pipeline = rs.pipeline()
config = rs.config()

# set color frame params
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.rotate(color_image,cv2.ROTATE_90_COUNTERCLOCKWISE)

        if not color_frame:
            break
        width, height = color_image.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(color_image)
            if_init = True

            ann_frame_idx = 0  # the frame index we interact with
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

            # Let's add a positive click at (x, y) = (210, 350) to get started
            points = np.array([[850, 380],[850, 220],[850, 540],[600, 226],[600, 400],[600, 600],[1050, 400]], dtype=np.float32)

            # for labels, `1` means positive click and `0` means negative click
            labels = np.array([1,1,1,0,0,0,0], np.int32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                                                                        frame_idx=ann_frame_idx,
                                                                        obj_id=ann_obj_id,
                                                                        points=points,
                                                                        labels=labels,)

        else:
            out_obj_ids, out_mask_logits = predictor.track(color_image)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            print(all_mask.shape)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
            color_image = cv2.addWeighted(color_image, 0.5, all_mask, 0.8, 0)

        cv2.imshow("color frame", color_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
