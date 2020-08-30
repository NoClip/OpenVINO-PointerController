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

log.basicConfig(level=log.INFO)


def init_vars():
    args = build_argparser().parse_args()

    global face_model, gaze_model, head_model, landmarks_model, device, input_type, input_file, threshold, benchmark, preview

    face_model = args.facemodel
    gaze_model = args.gazemodel
    head_model = args.headmodel
    landmarks_model = args.landmarksmodel

    device = args.device
    input_type = args.inputtype.lower()
    input_file = args.inputfile
    threshold = args.threshold
    benchmark = args.benchmark
    preview = args.preview

def start_infer():
    feed = None
    key_pressed = None

    if input_type == "cam":
        feed = InputFeeder(input_type="cam")
    else:
        if not os.path.isfile(input_file):
            log.error("cannot find file {}".format(input_file))
            exit(1)

        feed = InputFeeder(input_type="video", input_file=input_file)

    face_network = FaceDetectionModel(
        model_name=face_model, device=device, threshold=threshold
    )

    head_network = HeadPoseModel(
        model_name=head_model, device=device, threshold=threshold
    )

    landmarks_network = FacialLandmarksModel(
        model_name=landmarks_model, device=device, threshold=threshold
    )

    gaze_network = GazeModel(model_name=gaze_model, device=device, threshold=threshold)

    mouse_control = MouseController("medium", "fast")

    face_network.load_model()
    head_network.load_model()
    landmarks_network.load_model()
    gaze_network.load_model()

    feed.load_data()

    for flag, frame in feed.next_batch():
        if not flag:
            break

        if not benchmark:
            key_pressed = cv2.waitKey(60)

        face_output, cropped_face_frame = face_network.predict([frame])
        head_output, cropped_face_frame = head_network.predict([cropped_face_frame])
        landmarks_output, cropped_eyes = landmarks_network.predict([cropped_face_frame])
        mouse_coords, gaze_output = gaze_network.predict([head_output, cropped_eyes[0], cropped_eyes[1]])

        # disable preview and mouse control while benchmarking
        # to make it more accurate
        if not benchmark:
            # Input user from preview argument
            if preview:
                cv2.imshow("preview", cropped_face_frame)

            # using a dummy try/catch to prevent PyAutoGUI messeges 
            # when mouse reaches the screen edge
            try:
                mouse_control.move(mouse_coords[0], mouse_coords[1])
            except:
                pass

            if key_pressed == 27:
                break
    
    # save benchmarks values to output directory
    if benchmark:
        face_network.print_benchmark()
        head_network.print_benchmark()
        landmarks_network.print_benchmark()
        gaze_network.print_benchmark()

    cv2.destroyAllWindows()
    feed.close()

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-fm",
        "--facemodel",
        type=str,
        default=r"models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",
        help="Path to face detection model "
        "(default is models/face-detection-adas-binary-0001models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001)",
    )

    parser.add_argument(
        "-gm",
        "--gazemodel",
        type=str,
        default=r"models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002",
        help="Path to gaze estimation model "
        "(default is models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002)",
    )

    parser.add_argument(
        "-hm",
        "--headmodel",
        type=str,
        default=r"models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001",
        help="Path to head pose estimation model "
        "(default is models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001)",
    )

    parser.add_argument(
        "-lm",
        "--landmarksmodel",
        type=str,
        default=r"models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009",
        help=r"Path to landmarks regression model "
        "(default is models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009)",
    )

    parser.add_argument(
        "-if",
        "--inputfile",
        type=str,
        default="bin/demo.mp4",
        help=r"Path to video file " "(default is bin/demo.mp4)",
    )

    parser.add_argument(
        "-it",
        "--inputtype",
        type=str,
        default="file",
        help=r"input type (file or cam) " "(file by default)",
    )

    parser.add_argument(
        "-l",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help=r"MKLDNN (CPU)-targeted custom layers."
        "Absolute path to a shared library with the kernels impl. "
        "(None by default)",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help=r"Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help=r"Probability threshold for detections filtering " "(0.5 by default)",
    )

    parser.add_argument(
        "-bm",
        "--benchmark",
        type=bool,
        default=False,
        help=r"Run benchmark and calculate Model load time, Input/Output processing time and Inference time for each model "
        "(False by default)",
    )

    parser.add_argument(
        "-p",
        "--preview",
        type=bool,
        default=True,
        help=r"Show frames preview while predection " "(True by default)",
    )

    return parser

def main():
    init_vars()
    start_infer()

if __name__ == "__main__":
    main()
