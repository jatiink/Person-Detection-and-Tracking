import argparse
import os
import cv2
import torch
from numpy import random
# from torchvision import models
# import torch.nn as nn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, \
                non_max_suppression, \
                scale_coords,set_logging
from utils.torch_utils import TracedModel
from sort import *

############################################################################################
################################## Face Mask check model ###################################

# maskmodel = models.resnet34()
# num_ftrs = maskmodel.fc.in_features
# maskmodel.fc = nn.Linear(num_ftrs, 2)
# maskmodel.load_state_dict(torch.load('mask_check.pt', map_location=torch.device('cpu')))
# maskmodel.eval()

# def prediction(image):
#     try:
#         img = maskcheck.test_tranform(image)
#         with torch.no_grad():
#             new_prediction = model(img.view(1, 3, 224, 224))
#             return maskcheck.classes_names[new_prediction.argmax()]
#     except Exception as e:
#         print(e)

############################################################################################     
    
"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None, mask_check=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        if mask_check is not None:
            label = str(id) + ": " + str(mask_check)
        else:
            label = str(id) + ": " + str(names[cat])
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,0,0), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, [255, 255, 255], 1)
    return img

def detect():
    source = opt.source
    view_img = opt.view_img
    imgsz = 640
    trace = True
    

    # Initialize
    set_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load("yolov7.pt", map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device)

    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    mask_det_dict = {}
    
    if opt.save:
        if ".mp4" in opt.source:
            save_path = os.getcwd() + "\\newfile.mp4"
        else:  # 'video' or 'stream'
            save_path = os.getcwd() + "\\newfile.jpg"
    video_path = opt.source
    video = cv2.VideoCapture(video_path)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, classes=0)

        # Process detections
        for det in pred:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if len(det) :
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    if "person" not in s:
                        cv2.putText(im0, "No Pesron", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 1, lineType=cv2.LINE_AA)
                    if c == 0:
                        dets_to_sort = np.empty((0,6))
                        # NOTE: We send in detected object class too
                        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                            # if int(detclass) == 0:
                            dets_to_sort = np.vstack((dets_to_sort, 
                                        np.array([x1, y1, x2, y2, conf, detclass])))
                                        # image = im0[:, :, 3]
                        tracked_dets = sort_tracker.update(dets_to_sort)
                        tracks =sort_tracker.getTrackers()
                        for track in tracks:
                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                        int(track.centroidarr[i][1])), 
                                        (int(track.centroidarr[i+1][0]),
                                        int(track.centroidarr[i+1][1])),
                                        (255,0,0), thickness=2) 
                                        for i,_ in  enumerate(track.centroidarr) 
                                        if i < len(track.centroidarr)-1]
                            
                        # draw boxes for visualization
                        if len(tracked_dets)>0:
                            bbox_xyxy = tracked_dets[:,:4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            
                            for iter, box in enumerate(bbox_xyxy):
                                person_id = str(identities[iter])
                                if person_id in mask_det_dict:
                                    continue
                                x1, y1, x2, y2 = [int(i) for i in box]
                                poi = im0[y1:y2, x1:x2]
                                # mask_check = prediction(poi)
                                # mask_det_dict[person_id] = mask_check
                            draw_boxes(im0, bbox_xyxy, identities, categories, names)
            
            if view_img:
                cv2.imshow(str(p), im0)
            
            # Save results (image with detections)
              # increment run
            if opt.save:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:
                    vid_writer.write(im0)

        if cv2.waitKey(20) & 0xFF == 27:
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=r"C:\Users\jatin\Downloads\Tensor go assignment .mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--view-img', action='store_true', default=True, help='display results')
    parser.add_argument('--save', action='store_true', default=True, help='save images/videos')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()

    sort_tracker = Sort(max_age=5,
                       min_hits=2,
                       iou_threshold=0.2) 

    with torch.no_grad():
        detect()