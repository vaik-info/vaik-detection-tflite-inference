from typing import List, Dict, Tuple, Union
import multiprocessing
import platform
import math
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

from vaik_pascal_voc_rw_ex import pascal_voc_rw_ex


class TfliteModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None, num_thread: int = None):
        self.classes = classes
        num_thread = multiprocessing.cpu_count() if num_thread is None else num_thread
        self.__load(input_saved_model_path, num_thread)

    def inference(self, input_image: np.ndarray, score_th: float = 0.2, nms_th: Union[float, None] = 0.5) -> Tuple[List[Dict], Dict]:
        resized_image, resized_scale = self.__preprocess_image(input_image, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image)
        raw_pred = self.__raw_pred_parse(raw_pred)
        filter_pred = self.__filter_score(raw_pred, score_th)
        if nms_th is not None:
            filter_pred = self.__filter_nms(filter_pred, nms_th)
        objects_dict_list = self.__output_parse(filter_pred, resized_scale, input_image.shape[0:2])
        return objects_dict_list, raw_pred

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]), dtype=input_image.dtype)
        resized_scale = min(resize_input_shape[1] / input_image.shape[1],
                            resize_input_shape[0] / input_image.shape[0])
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image, (resize_input_shape[1] / resized_scale, resize_input_shape[0] / resized_scale)

    def __inference(self, resized_image: np.ndarray) -> List[np.ndarray]:
        if len(resized_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resized_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resized_image.dtype}')
        self.__set_input_tensor(resized_image)
        self.interpreter.invoke()
        raw_pred = self.__get_output_tensor()
        return raw_pred

    def __set_input_tensor(self, image: np.ndarray):
        input_tensor = self.interpreter.tensor(self.interpreter.get_input_details()[0]['index'])()
        input_tensor.fill(0)
        input_image = image.astype(self.interpreter.get_input_details()[0]['dtype'])
        input_tensor[0, :input_image.shape[0], :input_image.shape[1], :input_image.shape[2]] = input_image

    def __get_output_tensor(self) -> List[np.ndarray]:
        output_details = self.interpreter.get_output_details()
        output_tensor = []
        for index in range(len(output_details)):
            output = self.interpreter.get_tensor(output_details[index]['index'])
            scale, zero_point = output_details[index]['quantization']
            if scale > 1e-4:
                output = scale * (output - zero_point)
            output_tensor.append(output)
        return output_tensor

    def __raw_pred_parse(self, raw_pred: List[np.ndarray]):
        filter_pred = {}
        detection_classes_or_detection_scores = [pred[0] for pred in raw_pred if len(pred.shape) == 2]
        detection_classes_or_detection_scores = sorted(detection_classes_or_detection_scores,
                                                       key=lambda x: sum(np.modf(x)[0]))
        filter_pred['detection_boxes'] = [pred for pred in raw_pred if len(pred.shape) == 3][0][0]
        filter_pred['detection_classes'] = detection_classes_or_detection_scores[0]
        filter_pred['detection_scores'] = detection_classes_or_detection_scores[1]
        filter_pred['num_detections'] = [pred for pred in raw_pred if len(pred.shape) == 1][0][0]
        return filter_pred

    def __filter_score(self, pred, score_th):
        mask = pred['detection_scores'] > score_th
        filter_pred = {}
        filter_pred['detection_boxes'] = pred['detection_boxes'][mask]
        filter_pred['detection_classes'] = pred['detection_classes'][mask]
        filter_pred['detection_scores'] = pred['detection_scores'][mask]
        filter_pred['num_detections'] = int(filter_pred['detection_classes'].shape[0])
        return filter_pred

    def __calc_iou(cls, source_array, dist_array, source_area, dist_area):
        x_min = np.maximum(source_array[0], dist_array[:, 0])
        y_min = np.maximum(source_array[1], dist_array[:, 1])
        x_max = np.minimum(source_array[2], dist_array[:, 2])
        y_max = np.minimum(source_array[3], dist_array[:, 3])
        w = np.maximum(0, x_max - x_min + 0.0000001)
        h = np.maximum(0, y_max - y_min + 0.0000001)
        intersect_area = w * h
        iou = intersect_area / (source_area + dist_area - intersect_area)
        return iou

    def __filter_nms(cls, pred, nms_th):
        bboxes = pred['detection_boxes']
        areas = ((bboxes[:, 2] - bboxes[:, 0] + 0.0000001) * (bboxes[:, 3] - bboxes[:, 1] + 0.0000001))
        sort_index = np.argsort(pred['detection_scores'])
        i = -1
        while (len(sort_index) >= 2 - i):
            max_scr_ind = sort_index[i]
            ind_list = sort_index[:i]
            iou = cls.__calc_iou(bboxes[max_scr_ind], bboxes[ind_list], areas[max_scr_ind], areas[ind_list])
            del_index = np.where(iou >= nms_th)
            sort_index = np.delete(sort_index, del_index)
            i -= 1
        filter_pred = {}
        filter_pred['detection_boxes'] = pred['detection_boxes'][sort_index][::-1]
        filter_pred['detection_classes'] = pred['detection_classes'][sort_index][::-1]
        filter_pred['detection_scores'] = pred['detection_scores'][sort_index][::-1]
        filter_pred['num_detections'] = int(filter_pred['detection_classes'].shape[0])
        return filter_pred

    def __output_parse(self, pred: Dict, resized_scales: Tuple[int, int], image_shape: Tuple[int, int]) -> List[Dict]:
        objects_dict_list = []
        for pred_index in range(pred['num_detections']):
            classes_index = int(pred['detection_classes'][pred_index])
            name = str(classes_index) if self.classes is None else self.classes[classes_index]
            ymin = max(0, int((pred['detection_boxes'][pred_index][0] * resized_scales[0])))
            xmin = max(0, int((pred['detection_boxes'][pred_index][1] * resized_scales[1])))
            ymax = min(image_shape[0] - 1,
                       int((pred['detection_boxes'][pred_index][2] * resized_scales[0])))
            xmax = min(image_shape[1] - 1,
                       int((pred['detection_boxes'][pred_index][3] * resized_scales[1])))
            object_extend_dict = {'score': pred['detection_scores'][pred_index]}
            objects_dict = pascal_voc_rw_ex.get_objects_dict_template(name, xmin, ymin, xmax, ymax,
                                                                      object_extend_dict=object_extend_dict)
            objects_dict_list.append(objects_dict)
        return objects_dict_list

    def __load(self, input_saved_model_path: str, num_thread: int):
        try:
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, num_threads=num_thread)
            self.interpreter.allocate_tensors()
        except RuntimeError:
            _EDGETPU_SHARED_LIB = {
                'Linux': 'libedgetpu.so.1',
                'Darwin': 'libedgetpu.1.dylib',
                'Windows': 'edgetpu.dll'
            }[platform.system()]
            delegates = [tflite.load_delegate(_EDGETPU_SHARED_LIB)]
            self.interpreter = tflite.Interpreter(model_path=input_saved_model_path, experimental_delegates=delegates,
                                                  num_threads=num_thread)
            self.interpreter.allocate_tensors()
        self.model_input_shape = self.interpreter.get_input_details()[0]['shape']