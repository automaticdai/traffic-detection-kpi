#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ultralytics')


# In[ ]:


import cv2

from ultralytics import solutions

def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(300, 570), (900, 550), (750, 150), (500, 150)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count, tracker="bytetrack.yaml")

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        #cv2_imshow(results)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("../trafficData/four lanes.mp4", "trackzone.avi", "yolo11n.pt", [0, 2, 3, 5, 7])

