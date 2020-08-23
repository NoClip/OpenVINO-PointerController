# Computer Pointer Controller

This project translates eyes' gaze direction (from image, video or cam) to mouse pointer movement on computer, using Intel's OpenVINO Toolkit with four model:

- [Face detection model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html): Detects the face from the input (image, video or cam), cropping it and pass it for other models.
- [Head pose model](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html): Detects the head pose movement and direction.
- [Facial landmarks model](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html): Detects the left and right eyes.
- [Gaze estimation model](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html): Gets the head pose and eyes images from other models, and detects the eyes' gaze direction.

## Project Set Up and Installation
- Install Intel OpenVino version 2020.4.287 or later, please follow Intels's guide based on OS:  
-[Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)  
-[Windows FPGA](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows_fpga.html)  
-[Windows Docker](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_windows.html)  
-[Linux](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)  
-[Linux FPGA](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux_fpga.html)  
-[Linux Docker](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html)  
-[macOS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_macos.html)  
-[Raspbian* OS](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html)

- Create and activate Virtual Enviroment
  ```
  python3 -m venv pointer_env
  source pointer_env/bin/activate
  ```
- Install requirements from requirements.txt
  ```
  pip install -r requirements.txt
  ```
- Run OpenVINO's environment setup
  ```
  source /opt/intel/openvino/bin/setupvars.sh
  ```
- Download all models to `models/` directory is prefered
  ```
  python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
  --name landmarks-regression-retail-0009 -o models

  python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
  --name head-pose-estimation-adas-0001 -o models

  python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
  --name face-detection-adas-binary-0001 -o models

  python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
  --name gaze-estimation-adas-0002 -o models
  ```
- Directory structure should be as follows:
  ```
  ├── bin
  │   └── demo.mp4
  ├── models
  │   └── intel
  │       ├── face-detection-adas-binary-0001
  │       │   └── FP32-INT1
  │       │       ├── face-detection-adas-binary-0001.bin
  │       │       └── face-detection-adas-binary-0001.xml
  │       ├── gaze-estimation-adas-0002
  │       │   ├── FP16
  │       │   │   ├── gaze-estimation-adas-0002.bin
  │       │   │   └── gaze-estimation-adas-0002.xml
  │       │   ├── FP16-INT8
  │       │   │   ├── gaze-estimation-adas-0002.bin
  │       │   │   └── gaze-estimation-adas-0002.xml
  │       │   └── FP32
  │       │       ├── gaze-estimation-adas-0002.bin
  │       │       └── gaze-estimation-adas-0002.xml
  │       ├── head-pose-estimation-adas-0001
  │       │   ├── FP16
  │       │   │   ├── head-pose-estimation-adas-0001.bin
  │       │   │   └── head-pose-estimation-adas-0001.xml
  │       │   ├── FP16-INT8
  │       │   │   ├── head-pose-estimation-adas-0001.bin
  │       │   │   └── head-pose-estimation-adas-0001.xml
  │       │   └── FP32
  │       │       ├── head-pose-estimation-adas-0001.bin
  │       │       └── head-pose-estimation-adas-0001.xml
  │       └── landmarks-regression-retail-0009
  │           ├── FP16
  │           │   ├── landmarks-regression-retail-0009.bin
  │           │   └── landmarks-regression-retail-0009.xml
  │           ├── FP16-INT8
  │           │   ├── landmarks-regression-retail-0009.bin
  │           │   └── landmarks-regression-retail-0009.xml
  │           └── FP32
  │               ├── landmarks-regression-retail-0009.bin
  │               └── landmarks-regression-retail-0009.xml
  ├── pointer_env
  │   └── ...  
  ├── pipeline.png
  ├── README.md
  ├── requirements.txt
  └── src
      ├── face_detection.py
      ├── facial_landmarks_detection.py
      ├── gaze_estimation.py
      ├── head_pose_estimation.py
      ├── input_feeder.py
      ├── main.py
      ├── model.py
      └── mouse_controller.py
  ```

## Demo
Run the following without any parameters will assume the defaults accoring to the table in the documentation section:
  ```
  python3 src/main.py
  ```

## Documentation

| Parameter  | Description            | Default value                                                                          |
| -----------|------------------------| ---------------------------------------------------------------------------------------|
| -fm        | Face detection model   | models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 |
| -gm        | Gaze estimation model  | models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002                  |
| -hm        | Head pose model        | models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001        |
| -lm        | Facial landmarks model | models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009    |
| -if        | Input File path        | bin/demo.mp4                                                                           |
| -it        | Input Type (File/CAM)  | file                                                                                   |
| -l         | Extensions             | None                                                                                   |
| -d         | Device                 | CPU                                                                                    |
| -t         | Threshold              | 0.5                                                                                    |

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
