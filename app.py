from icevision.all import *
import icedata
import PIL, requests
import torch
from torchvision import transforms
import gradio as gr

# Download the dataset
url = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
dest_dir = "fridge"
data_dir = icedata.load_data(url, dest_dir)

# Create the parser
parser = parsers.VOCBBoxParser(annotations_dir=data_dir / "odFridgeObjects/annotations", images_dir=data_dir / "odFridgeObjects/images")

# Parse annotations to create records
train_records, valid_records = parser.parse()

class_map = parser.class_map

extra_args = {}
model_type = models.torchvision.retinanet
backbone = model_type.backbones.resnet50_fpn
# Instantiate the model
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), **extra_args) 

# Transforms
# size is set to 384 because EfficientDet requires its inputs to be divisible by 128
image_size = 384
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=image_size, presize=512), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])
# Datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)
# Data Loaders
train_dl = model_type.train_dl(train_ds, batch_size=8, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=8, num_workers=4, shuffle=False)
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

learn = learn.load('model')

import os
for root, dirs, files in os.walk(r'sample_images/'):
    for filename in files:
        print(filename)

examples = ["sample_images/"+file for file in files] 
article="<p style='text-align: center'><a href='https://dicksonneoh.com/fridge-detector/' target='_blank'>Blog post</a></p>"
enable_queue=True



def show_preds(input_image, display_label, display_bbox, detection_threshold):

    if detection_threshold==0: detection_threshold=0.5

    img = PIL.Image.fromarray(input_image, 'RGB')

    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=detection_threshold,
                                           display_label=display_label, display_bbox=display_bbox, return_img=True, 
                                           font_size=16, label_color="#FF59D6")

    return pred_dict['img']

# display_chkbox = gr.inputs.CheckboxGroup(["Label", "BBox"], label="Display", default=True)
display_chkbox_label = gr.inputs.Checkbox(label="Label", default=True)
display_chkbox_box = gr.inputs.Checkbox(label="Box", default=True)

detection_threshold_slider = gr.inputs.Slider(minimum=0, maximum=1, step=0.1, default=0.5, label="Detection Threshold")

outputs = gr.outputs.Image(type="pil")

# Option 1: Get an image from local drive
gr_interface = gr.Interface(fn=show_preds, inputs=["image", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - Fridge Object', article=article,examples=examples)

# #  Option 2: Grab an image from a webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=False)

# #  Option 3: Continuous image stream from the webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=True)


gr_interface.launch(inline=False, share=True, debug=True)
