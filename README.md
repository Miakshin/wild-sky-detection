# Drone Detection

This repository contains code and resources for detecting drones using machine learning techniques.
The project aims to identify drones in various environments using image and video data.
This is a beginning of a project that has an aim to be run on edge devices for real-time drone detection.

The model for object detection is based on YOLOv5s and trained on a custom dataset containing images of drones, birds, and fixed objects.
Model was trained using Ultralytics YOLOv8 framework in Google Colab. Data for dataset was collected from roboflow and other open sources.

## Features
- Real-time drone/bird/fixed-wing detection using pre-trained models
- Support for image and video input

## Examples
![Drone Detection Example](examples/video_1.gif)
![Drone Detection Example](examples/video_2.gif)
![Drone Detection Example](examples/video_3.gif)

# Usage
