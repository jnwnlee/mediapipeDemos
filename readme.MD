# Mediapipe examples

From:
https://google.github.io/mediapipe/


## Installation

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For the iris example, download `iris_landmark.tflite` into `models` directory:
```
wget -P models https://github.com/google/mediapipe/raw/master/mediapipe/modules/iris_landmark/iris_landmark.tflite
```

## How to run

One of the following:

```
python face_detection.py
python face_mesh.py
python hands.py
python head_posture.py
python holistic.py
python iris.py
python objectron.py
python pose.py
python selfie_segmentation.py
```
+ face_filter.py , filter_gradio.py, head_posture added/modified by [jnwnlee](https://github.com/jnwnlee).

`pose.py` and `iris.py` include the possibility to process a video file instead of the webcam input stream. Run like this:

```
python iris.py -i /path/to/some/file/i-am-a-video-file.mp4
python pose.py -i /path/to/some/file/i-am-a-video-file.mp4
```

## Numpy

See example `python pose.py` for how to extract numpy array from the mediapipe landmark objects.
