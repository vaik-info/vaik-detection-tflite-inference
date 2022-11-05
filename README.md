# vaik-detection-tflite-inference

Inference with the Tflite model of the Tensorflow Object Detection API and output the result as a dict in extended Pascal VOC format.

## Example

![vaik-detection-tflite-inference](https://user-images.githubusercontent.com/116471878/200096718-316a1a8c-7033-42f1-9358-1643489c89f8.png)

## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-detection-tflite-inference.git
```

## Usage

### Example

```python
import os
import numpy as np
from PIL import Image

from vaik_detection_tflite_inference.tflite_model import TfliteModel

input_saved_model_path = os.path.expanduser('~/export_model/efficientdet-lite0_edgetpu.tflite')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-detection-dataset/valid/valid_000000000.jpg')).convert('RGB'))

model = TfliteModel(input_saved_model_path, classes, num_thread=8)
objects_dict_list, raw_pred = model.inference(image, score_th=0.2, nms_th=0.5)
```

#### Output

- objects_dict_list

```text
[
  {
    'name': 'eight',
    'pose': 'Unspecified',
    'truncated': 0,
    'difficult': 0,
    'bndbox': {
      'xmin': 564,
      'ymin': 100,
      'xmax': 611,
      'ymax': 185
    },
    'score': 0.9445509314537048
  },
  ・・・
  {
    'name': 'four',
    'pose': 'Unspecified',
    'truncated': 0,
    'difficult': 0,
    'bndbox': {
      'xmin': 40,
      'ymin': 376,
      'xmax': 86,
      'ymax': 438
    },
    'score': 0.38432005047798157
  }
]
```

- raw_pred
```
{'detection_boxes': array([[ 4.87580299e-01,  3.82389069e-01,  5.87419748e-01,
       ・・・
      dtype=float32), 'num_detections': 100.0}

```