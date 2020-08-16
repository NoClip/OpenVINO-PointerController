from openvino.inference_engine import IENetwork, IECore
import os
import cv2
from argparse import ArgumentParser
import logging as log
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksModel
from gaze_estimation import GazeModel
from head_pose_estimation import HeadPoseModel
from input_feeder import InputFeeder
from mouse_controller import MouseController

# os.chdir(
#     r"C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\python\python3.7")


# import sys
# import time
# import socket
# import json


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-fm", "--facemodel", type=str,
                        default="models\face-detection-adas-binary-0001",
                        help="Path to face detection model, default is models\\face-detection-adas-binary-0001")

    parser.add_argument("-gm", "--gazemodel", type=str,
                        default="models\gaze-estimation-adas-0002",
                        help="Path to gaze estimation model, default is models\\gaze-estimation-adas-0002")

    parser.add_argument("-hm", "--headmodel", type=str,
                        default="models\head-pose-estimation-adas-0001",
                        help="Path to head pose estimation model, default is models\\head-pose-estimation-adas-0001")

    parser.add_argument("-lm", "--landmarksmodel", type=str,
                        default="models\landmarks-regression-retail-0009",
                        help="Path to landmarks regression model, default is models\\landmarks-regression-retail-0009")

    parser.add_argument("-i", "--input", type=str,
                        help="Path to video file, default is bin\demo.mp4")

    parser.add_argument("-is", "--inputsource", type=str,
                        help="input source (file or cam), default is file")

    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")

    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")

    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def main():
    args = build_argparser().parse_args()

    model = args.model
    device = args.device
    video_file = args.video
    max_people = args.max_people
    threshold = args.threshold
    output_path = args.output_path


if __name__ == '__main__':
    main()
