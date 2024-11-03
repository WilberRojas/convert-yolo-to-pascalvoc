#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:10:01 2018

@author: Caroline Pacheco do E. Silva
"""

import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join

class YOLO2PascalVOC:
    def __init__(self, classes: tuple, root: str):
        self.classes = classes
        self.root = root
        
        self.images_folder = 'images'
        self.labels_folder = 'labels'  # YOLO Labels folder
        self.output_folder = "labels_PascalVOC" # PascalVOC folder

    def unconvert(self, class_id, width, height, x, y, w, h):
        xmax = int((x * width) + (w * width) / 2.0)
        xmin = int((x * width) - (w * width) / 2.0)
        ymax = int((y * height) + (h * height) / 2.0)
        ymin = int((y * height) - (h * height) / 2.0)
        class_id = int(class_id)
        return (class_id, xmin, xmax, ymin, ymax)

    def run(self):
        yolo_labels_path = join(self.root, self.labels_folder)
        ids = list()
        list_dir = os.listdir(yolo_labels_path)

        ids = [x.split(".")[0] for x in list_dir]

        annopath = join(self.root, self.labels_folder, "%s.txt")
        imgpath = join(self.root, self.images_folder,"%s.jpg")

        os.makedirs(join(self.root, self.output_folder), exist_ok=True)
        outpath = join(self.root, self.output_folder, "%s.xml")

        for i in range(len(ids)):
            img_id = ids[i]
            img = cv2.imread(imgpath % img_id)
            height, width, channels = img.shape

            node_root = Element("annotation")
            node_folder = SubElement(node_root, "folder")
            node_folder.text = "VOC2007"
            img_name = img_id + ".jpg"

            node_filename = SubElement(node_root, "filename")
            node_filename.text = img_name

            node_source = SubElement(node_root, "source")
            node_database = SubElement(node_source, "database")
            node_database.text = "Coco database"

            node_size = SubElement(node_root, "size")
            node_width = SubElement(node_size, "width")
            node_width.text = str(width)

            node_height = SubElement(node_size, "height")
            node_height.text = str(height)

            node_depth = SubElement(node_size, "depth")
            node_depth.text = str(channels)

            node_segmented = SubElement(node_root, "segmented")
            node_segmented.text = "0"

            target = annopath % img_id
            if os.path.exists(target):
                label_norm = np.loadtxt(target).reshape(-1, 5)

                for i in range(len(label_norm)):
                    labels_conv = label_norm[i]
                    new_label = self.unconvert(
                        labels_conv[0],
                        width,
                        height,
                        labels_conv[1],
                        labels_conv[2],
                        labels_conv[3],
                        labels_conv[4],
                    )
                    node_object = SubElement(node_root, "object")
                    node_name = SubElement(node_object, "name")
                    node_name.text = self.classes[new_label[0]]

                    node_pose = SubElement(node_object, "pose")
                    node_pose.text = "Unspecified"

                    node_truncated = SubElement(node_object, "truncated")
                    node_truncated.text = "0"
                    node_difficult = SubElement(node_object, "difficult")
                    node_difficult.text = "0"
                    node_bndbox = SubElement(node_object, "bndbox")
                    node_xmin = SubElement(node_bndbox, "xmin")
                    node_xmin.text = str(new_label[1])
                    node_ymin = SubElement(node_bndbox, "ymin")
                    node_ymin.text = str(new_label[3])
                    node_xmax = SubElement(node_bndbox, "xmax")
                    node_xmax.text = str(new_label[2])
                    node_ymax = SubElement(node_bndbox, "ymax")
                    node_ymax.text = str(new_label[4])
                    xml = tostring(node_root, pretty_print=True)
                    parseString(xml)

            f = open(outpath % img_id, "wb")
            f.write(xml)
            f.close()

if __name__ == "__main__":
    # Usage:
    YOLO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush')

    converter = YOLO2PascalVOC(
        classes=YOLO_CLASSES,
        root=".../dataset" ## path root folder
    )

    converter.run()
