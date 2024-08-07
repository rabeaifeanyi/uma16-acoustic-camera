import cv2

from bokeh.plotting import curdoc
from bokeh.layouts import column

from ui import *
from processing import *
from config import *


VIDEO_INDEX = 0
MIC_INDEX = 0
VIDEO_SCALE_FACTOR = 1


vc = cv2.VideoCapture(VIDEO_INDEX)
frame_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH) * VIDEO_SCALE_FACTOR)
frame_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT) * VIDEO_SCALE_FACTOR)

config = ConfigUMA()

video_stream = VideoStream(frame_width, frame_height, vc)
model_processor = ModelProcessor(frame_width, frame_height, config)


fig, cds, cameraCDS = create_plot(frame_width, frame_height)


def update_camera_view():
    img = video_stream.get_frame()
    if img is not None:
        cameraCDS.data['image_data'] = [img]

def update_estimations():
    model_data = model_processor.get_uma16_dummy_data()
    update_plot(cds, model_data)
    


doc = curdoc()
doc.add_periodic_callback(update_estimations, 1000)
doc.add_periodic_callback(update_camera_view, 100)
doc.add_root(column(fig))
