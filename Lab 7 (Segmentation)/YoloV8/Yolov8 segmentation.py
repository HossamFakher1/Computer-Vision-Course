import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolov8n-seg.pt")   # best (1).pt
cap = cv2.VideoCapture(0) # traffictrim.mp4
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('instance-segmentation-object-tracking.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print(" Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0 ,iou=0.5, show=False ,persist=True , tracker="bytetrack.yaml")

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask,
                                mask_color=colors(track_id, True),
                                track_label=str(track_id))
            # cv2.polylines(im0, [np.int32([mask])], isClosed=True, color=(255,0,0), thickness=2)
            # cv2.putText(im0, f'{track_id}', (int(mask[0][0]),int(mask[0][1])), cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0), 3)

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()