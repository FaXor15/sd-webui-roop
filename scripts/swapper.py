import copy
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import insightface
import onnxruntime
from scripts.cimage import convert_to_sd

from modules.face_restoration import FaceRestoration, restore_faces
from modules.upscaler import Upscaler, UpscalerData
from scripts.roop_logging import logger

providers = ["CPUExecutionProvider"]


@dataclass
class UpscaleOptions:
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def upscale_image(image: Image, upscale_options: UpscaleOptions):
    result_image = image
    if upscale_options.upscaler is not None and upscale_options.upscaler.name != "None":
        original_image = result_image.copy()
        logger.info(
            "Upscale with %s scale = %s",
            upscale_options.upscaler.name,
            upscale_options.scale,
        )
        result_image = upscale_options.upscaler.scaler.upscale(
            image, upscale_options.scale, upscale_options.upscaler.data_path
        )
        if upscale_options.scale == 1:
            result_image = Image.blend(
                original_image, result_image, upscale_options.upscale_visibility
            )

    if upscale_options.face_restorer is not None:
        original_image = result_image.copy()
        logger.info("Restore face with %s", upscale_options.face_restorer.name())
        numpy_image = np.array(result_image)
        numpy_image = upscale_options.face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            original_image, restored_image, upscale_options.restorer_visibility
        )

    return result_image


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None


def swap_all_faces(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    upscale_options: Union[UpscaleOptions, None] = None,
) -> List[ImageResult]:
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if model is not None and not scale:
        if isinstance(source_img, str):  
            import base64, io
            if 'base64,' in source_img:  
                base64_data = source_img.split('base64,')[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
        face_swapper = getFaceSwapModel(model_path)

        source_faces = []
        for face_num in range(len(faces_indices)):
            source_face = get_face_single(source_img, face_index=face_num)
            if source_face is not None:
                source_faces.append(source_face)
            else:
                logger.info(f"No source face found for face {face_num}")

        if not source_faces:
            logger.info("No source faces found in the input image.")
            return []

        restored_images = []

        for source_face in source_faces:
            result = target_img.copy()

            for target_face_num in range(len(source_faces)):
                if source_face == source_faces[target_face_num]:
                    continue
                target_face = get_face_single(target_img, face_index=target_face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    logger.info(f"No target face found for face {target_face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if upscale_options is not None:
                result_image = upscale_image(result_image, upscale_options)
            restored_images.append(result_image)

    return [ImageResult(path=fn.name, image=image) for image in restored_images]

