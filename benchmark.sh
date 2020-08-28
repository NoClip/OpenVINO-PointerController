rm output/*

python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
-gm models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml \
-hm models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml \
-lm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
-bm True

python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
-gm models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml \
-hm models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml \
-lm models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
-bm True

python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \
-gm models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml \
-hm models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml \
-lm models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml \
-bm True

python3 src/plot_benchmark.py