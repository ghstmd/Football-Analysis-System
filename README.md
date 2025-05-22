# Football-Analysis-System

## This project was developed as part of the "Thực Tập Cơ Sở" (Fundamental Internship) course at the Posts and Telecommunications Institute of Technology (PTIT), Artificial Intelligence Department, in December 2024.

## Introduction
This project focuses on analyzing football match footage by detecting and tracking players, referees, and the ball using the YOLO object detection model. To enhance detection accuracy, the model is further trained on custom data. Players are grouped into teams based on the color of their jerseys using K-means clustering for pixel segmentation. This allows us to compute ball possession percentages for each team throughout the match.

To track player movement more precisely, optical flow is used to compensate for camera motion between frames. Perspective transformation is applied to convert pixel coordinates into real-world measurements, enabling the calculation of distances in meters. With this data, the system estimates each player’s speed and total distance covered during the match.

![Screenshot](output_videos/screenshot1.png)
![Screenshot](output_videos/screenshot2.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Dataset
- DFL - Bundesliga Data Shootout, A competition dataset by Deutsche Fußball Liga e.V. on Kaggle.
- Football Players Detection Dataset from Roboflow.

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

