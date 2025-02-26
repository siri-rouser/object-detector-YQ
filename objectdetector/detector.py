import logging
import re
import time
from typing import Any, List, NamedTuple
import cv2
import numpy as np
import torch
from prometheus_client import Counter, Histogram, Summary
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from visionapi.messages_pb2 import Metrics, SaeMessage, VideoFrame
from visionlib.pipeline.tools import get_raw_frame_data

from .batch import BatchEntry
from .config import ObjectDetectorConfig
from functools import wraps

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('object_detector_get_duration', 'The time it takes to deserialize the proto until returning the detection result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))
MODEL_DURATION = Summary('object_detector_model_duration', 'How long the model call takes (without NMS)')
NMS_DURATION = Summary('object_detector_nms_duration', 'How long non-max suppression takes')
OBJECT_COUNTER = Counter('object_detector_object_counter', 'How many objects have been detected')
PROTO_SERIALIZATION_DURATION = Summary('object_detector_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('object_detector_proto_deserialization_duration', 'The time it takes to deserialize an input proto')

class ProcessEntry(NamedTuple):
    numpy_image: np.ndarray
    video_frame: VideoFrame

class Detector:
    def __init__(self, config: ObjectDetectorConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)

        self.model = None
        self.device = None
        self.input_image_size = None
        self.background = cv2.cuda.createBackgroundSubtractorMOG2(history=200, varThreshold=70, detectShadows=True)

        self._setup_model()

    def _setup_model(self):
        logger.info('Setting up object-detector model...')
        self.device = torch.device(self.config.model.device)
        self.model = AutoBackend(
            self._yolo_weights(),
            device=self.device,
            fp16=self.config.model.fp16_quantization
        )
        self.input_image_size = check_imgsz(self.config.inference_size, stride=self.model.stride)

    def _yolo_weights(self):
        weights_path = self.config.model.weights_path
        if weights_path.is_file():
            return self.config.model.weights_path
        elif re.match(r'^yolov8[nsmlx].pt$', weights_path.name) is not None and self.config.model.auto_download:
            return weights_path
        else:
            raise IOError(f'Could not load weights with current model config: {self.config.model.model_dump_json()}')

    def __call__(self, input_proto, *args, **kwargs) -> Any:
        return self.get(input_proto)

    @GET_DURATION.time()
    @torch.no_grad()
    def get(self, input_batch: List[BatchEntry]) -> List[BatchEntry]:
        process_batch: List[ProcessEntry] = []

        flag = 0
        for entry in input_batch:
            numpy_image, video_frame = self._unpack_proto(entry.proto_data)
            if self.config.motion_detect:
                if self._detect_motion(numpy_image):
                    prepared_image = self._prepare_input(numpy_image)
                    process_batch.append(ProcessEntry(prepared_image, video_frame))
                else:
                    logger.info('Skipping frame due to no motion detected')
                    prepared_image = np.zeros_like(numpy_image)
                    process_batch.append(ProcessEntry(prepared_image, video_frame))
                    flag+=1
                    continue
            else:
                prepared_image = self._prepare_input(numpy_image)
                process_batch.append(ProcessEntry(prepared_image, video_frame))
        
        if flag == len(input_batch):
            print('No motion detected in the batch')
            output_batch = []
            for idx,input_entry in enumerate(input_batch):
                output_batch.append(BatchEntry(input_entry.stream_key, self._create_output([], process_batch[idx].video_frame,0)))
            return output_batch
        else:
            numpy_batch = np.array([entry.numpy_image for entry in process_batch])
            numpy_batch_ct = np.ascontiguousarray(numpy_batch)
            batch_tensor = torch.from_numpy(numpy_batch_ct).to(self.device).float() / 255.0

            inference_start = time.time_ns()

            with MODEL_DURATION.time():
                yolo_prediction = self.model(batch_tensor)

            with NMS_DURATION.time():
                predictions = non_max_suppression(
                    yolo_prediction, 
                    conf_thres=self.config.model.confidence_threshold, 
                    iou_thres=self.config.model.iou_threshold,
                    classes=self.config.classes,
                    agnostic=self.config.model.nms_agnostic,
                )

            inference_time_us = (time.time_ns() - inference_start) // 1000

            output_batch = []
            for input_entry, process_entry, prediction in zip(input_batch, process_batch, predictions):
                prediction[:, :4] = scale_boxes(process_entry.numpy_image.shape[1:], prediction[:, :4], (process_entry.video_frame.shape.height, process_entry.video_frame.shape.width))
                self._normalize_boxes(prediction, (process_entry.video_frame.shape.height, process_entry.video_frame.shape.width))
                OBJECT_COUNTER.inc(len(prediction))
                output_batch.append(BatchEntry(input_entry.stream_key, self._create_output(prediction, process_entry.video_frame, inference_time_us // len(input_batch))))
            return output_batch


    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        input_image = get_raw_frame_data(sae_msg.frame)
        return input_image, sae_msg.frame
    
    # def _prepare_input(self, image) -> torch.Tensor:
    #     out_img = LetterBox(self.input_image_size, auto=True, stride=self.model.stride)(image=image)
    #     out_img = out_img.transpose((2, 0, 1))[::-1]
    #     print(out_img.shape)
    #     return out_img
    
    def _prepare_input(self, image):
        """Prepares image for TensorRT inference ensuring it is resized correctly to self.input_image_size."""
    
        # Ensure that input size is exactly as expected
        target_size = tuple(self.input_image_size)  # Convert list to tuple if necessary

        # Resize the image to match the expected input size
        out_img = LetterBox(target_size, auto=False, stride=32)(image=image)  

        # Convert HWC (Height-Width-Channel) to CHW (Channel-Height-Width)
        out_img = out_img.transpose((2, 0, 1))  
        return out_img

    def _normalize_boxes(self, predictions, image_shape):
        predictions[:,0] /= image_shape[1]
        predictions[:,2] /= image_shape[1]
        predictions[:,1] /= image_shape[0]
        predictions[:,3] /= image_shape[0]

    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, predictions, frame_proto, inference_time_us):
        sae_msg = SaeMessage()

        for pred in predictions:
            bb_min_x = float(pred[0])
            bb_min_y = float(pred[1])
            bb_max_x = float(pred[2])
            bb_max_y = float(pred[3])

            if self.config.drop_edge_detections and self._is_edge_bounding_box(bb_min_x, bb_min_y, bb_max_x, bb_max_y):
                continue

            detection = sae_msg.detections.add()

            detection.bounding_box.min_x = bb_min_x
            detection.bounding_box.min_y = bb_min_y
            detection.bounding_box.max_x = bb_max_x
            detection.bounding_box.max_y = bb_max_y

            detection.confidence = float(pred[4])
            detection.class_id = int(pred[5])

        sae_msg.frame.CopyFrom(frame_proto)

        sae_msg.metrics.detection_inference_time_us = inference_time_us

        return sae_msg.SerializeToString()

    def _is_edge_bounding_box(self, min_x: float, min_y: float, max_x: float, max_y: float) -> bool:
        return any((
            min_x < 0.01,
            min_y < 0.01,
            max_x > 0.99,
            max_y > 0.99
        ))
    
    def time_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            ms_time = 1000*(end_time - start_time)
            logger.info(f"{func.__name__} took {ms_time:.6f} ms")
            return result
        return wrapper

    @time_function
    def _detect_motion(self, image):
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        fg_mask = self.background.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Count non-zero pixels in fg_mask
        motion_pixels = cv2.countNonZero(fg_mask)
        if motion_pixels > self.config.motion_threshold:
            return True
        else:
            return False
