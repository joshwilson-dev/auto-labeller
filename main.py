################
#### Header ####
################

# Title: Automated Labeller
# Author: Josh Wilson
# Date: 07-07-2022
# Description: 
# This script uses an onnx detection model to create an
# annotation file images in a directory

###############
#### Setup ####
###############
import json
import os
from pickle import FALSE, TRUE
from PIL import Image
import onnxruntime
import numpy
import tkinter as tk
from tkinter import filedialog
import time

#################
#### Content ####
#################

# create function for user to select dir
root = tk.Tk()
root.title('Automated labeller')
canvas = tk.Canvas(root, width = 600, height = 300, bg='white')
canvas.grid(columnspan=3, rowspan = 3)

# logo
app_dir = os.path.dirname(os.path.abspath(__file__))
logo = tk.PhotoImage(file = os.path.join(app_dir, "logo.png"))
logo = logo.subsample(5)
logo_label = tk.Label(image=logo, borderwidth=0)
logo_label.image = logo
logo_label.grid(column=1, row=0)

# instructions
instructions = tk.Label(root, text="Select a folder containing the images you want to label", font = "Raleway", bg='white')
instructions.grid(columns=3, column=0, row=1)

def select_dir():
    folder = filedialog.askdirectory(title = "Select folder")
    if folder:
        duration = main(folder)
        browse_text.set("I'm done, I took {}s, click to try again".format(round(duration, 2)))

# browse button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=lambda:select_dir(), font="Raleway", bg="#153db3", fg="white", height=2, width=50)
browse_text.set("Let's see who can label faster")
browse_btn.grid(column=1, row=2)

# model
app_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(app_dir, "model_final_state_dict.onnx")
ort_session = onnxruntime.InferenceSession(model_path)
idx_to_class = {
    1: "senecio-lautus-seed-fertilised",
    2: "senecio-lautus-seed-unfertilised"
}

def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    width = image.width
    height = image.height
    CHW_image = numpy.array(image)
    tensor_image = numpy.array(numpy.expand_dims(CHW_image.transpose((2, 0, 1))/255,0), numpy.float32)
    return tensor_image, width, height

def label(image_path, image_name):
    image, width, height = prepare_image(os.path.join(image_path, image_name))
    ort_inputs = {ort_session.get_inputs()[0].name:image}
    prediction = ort_session.run(None, ort_inputs)
    boxes = prediction[0]
    labels = prediction[1]
    points = []
    for box in boxes:
        points.append([
            [float(box[0]), float(box[1])],
            [float(box[2]), float(box[3])]])
    named_labels = [idx_to_class[i] for i in labels]
    label_name = os.path.splitext(image_name)[0] + '.json'
    label_path = os.path.join(image_path, label_name)
    shapes = []
    for i in range(0, len(named_labels)):
        shapes.append({
            "label": named_labels[i],
            "points": points[i],
            "group_id": 'null',
            "shape_type": "rectangle",
            "flags": {}})
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": 'null',
        "imageHeight": height,
        "imageWidth": width}
    annotation_str = json.dumps(annotation, indent = 2).replace('"null"', 'null')
    with open(label_path, 'w') as annotation_file:
        annotation_file.write(annotation_str)

def main(file_path_variable):
    start_time = time.time()
    # walk through image files and label
    for file in os.listdir(file_path_variable):
        if file.endswith(".JPG"):
            label(file_path_variable, file)
    duration = time.time() - start_time
    return duration

root.mainloop()