import os
import cv2
import glob
import onnx
import torch
import ctypes
import argparse
import numpy as np
import logging
import pycuda.autoinit
from torch import nn
from tqdm import tqdm
import tensorrt as trt
import onnxruntime as rt
import torch.optim as optim
import pycuda.driver as cuda
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ImageFormatReader import ImageFormatReader
from torch.utils.model_zoo import load_url as load_state_dict_from_url


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT")
    parser.add_argument("--tile_size", help="tile size", default=2048, type=int)

    parser.add_argument("--batch_size", type=int, help="data batch size", default=1)
    parser.add_argument("--img_size", help="input size", default=[3, 2048, 2048])
    parser.add_argument(
        "--patch_folder",
        help="patches folder path",
        default="/home/amans/Desktop/artifact_data_copy/val/train/",
    )
    parser.add_argument(
        "--onnx_model_path",
        help="onnx model path",
        default="/mnt/prj003_new/users/tanya/fpn_original.onnx",
    )
    parser.add_argument(
        "--tensorrt_engine_path",
        help="tensorrt engine path",
        default="/mnt/prj003_new/users/tanya/fpn_original.engine",
    )

    # TensorRT engine params
    parser.add_argument(
        "--dynamic_axes", help="dynamic batch input or output", default="False"
    )
    parser.add_argument(
        "--engine_precision",
        help="precision of TensorRT engine",
        choices=["FP32", "FP16", "INT8"],
        default="INT8",
    )
    parser.add_argument(
        "--min_engine_batch_size",
        type=int,
        help="set the min input data size of model for inference",
        default=2,
    )
    parser.add_argument(
        "--opt_engine_batch_size",
        type=int,
        help="set the most used input data size of model for inference",
        default=1,
    )
    parser.add_argument(
        "--max_engine_batch_size",
        type=int,
        help="set the max input data size of model for inference",
        default=1,
    )
    parser.add_argument(
        "--engine_workspace", type=int, help="workspace of engine", default=1024
    )

    args = parser.parse_args()

    return args


class DataLoader:
    def __init__(self, path, tile_size, batch_size):
        self.index = 0
        self.path = path
        self.batch_size = batch_size
        self.root = sorted(os.listdir(self.path))
        self.length = 5000
        self.model_input_size = tile_size
        self.calibration_data = np.zeros(
            (batch_size, 3, tile_size, tile_size), dtype=np.float32
        )
        self.base = path
        self.all_wsi_paths = []
        for p in self.root:
            name = os.path.basename(p)
            name_part = name.split(".")
            if name_part[-1] == "png":
                if not (
                    (name_part[-2].endswith("_label"))
                    or (name_part[-2].endswith("_aug3"))
                ):
                    return_name = self.base + name
                    self.all_wsi_paths.append(return_name)

    def reset(self):
        batch_size = 1

    def next_batch(self):
        if self.index <= self.length:
            calibration_dat = self.all_wsi_paths[self.index]
            print(calibration_dat)
            thumb_nail = cv2.imread(calibration_dat)
            thumb_nail = cv2.cvtColor(thumb_nail, cv2.COLOR_BGR2RGB)
            trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            if not calibration_dat.endswith("_label.png"):
                thumb_nail = (trans(thumb_nail)).numpy()
            self.calibration_data = np.expand_dims(thumb_nail, axis=0)
            self.index += 1
            yield np.ascontiguousarray(self.calibration_data, dtype=np.float32)

    def __len__(self):
        return self.length


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, stream, cache_file, batch_size):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.stream = stream
        self.batch_size = batch_size
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        #self.calibration_data = np.zeros((batch_size, 3, self.tile_size, self.tile_size), dtype=np.float32)
        #self.d_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next((self.stream.next_batch()))
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]
        except StopIteration:
            return None
        except IndexError:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info(
                    "Using calibration cache to save time: {:}".format(self.cache_file)
                )
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info(
                "Caching calibration data for future use: {:}".format(self.cache_file)
            )
            f.write(cache)


def build_engine(
    onnx_model_path,
    tensorrt_engine_path,
    engine_precision,
    dynamic_axes,
    img_size,
    batch_size,
    min_engine_batch_size,
    opt_engine_batch_size,
    max_engine_batch_size,
    calibration_stream,
    calibration_table,
):
    # Builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Set precision
    if engine_precision == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")
    elif engine_precision == "INT8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(
            calibration_stream, calibration_table, batch_size
        )
        # print(config.int8_calibrator.get_batch())
        print("Int8 mode enabled")

    # Onnx parser
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_model_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    # Input
    inputTensor = network.get_input(0)
    # Dynamic batch (min, opt, max)
    print("inputTensor.name:", inputTensor.name)
    if dynamic_axes:
        profile.set_shape(
            inputTensor.name,
            (min_engine_batch_size, img_size[0], img_size[1], img_size[2]),
            (opt_engine_batch_size, img_size[0], img_size[1], img_size[2]),
            (max_engine_batch_size, img_size[0], img_size[1], img_size[2]),
        )
        print("Set dynamic")
    else:
        profile.set_shape(
            inputTensor.name,
            (batch_size, img_size[0], img_size[1], img_size[2]),
            (batch_size, img_size[0], img_size[1], img_size[2]),
            (batch_size, img_size[0], img_size[1], img_size[2]),
        )
    config.add_optimization_profile(profile)
    print(network.get_output(0).name)

    # Write engine
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engineString)


def main():
    args = parse_args()

    calibration_stream = DataLoader(args.patch_folder, args.tile_size, args.batch_size)

    calibration_table = "best_calibration.cache"

    build_engine(
        args.onnx_model_path,
        args.tensorrt_engine_path,
        args.engine_precision,
        args.dynamic_axes,
        args.img_size,
        args.batch_size,
        args.min_engine_batch_size,
        args.opt_engine_batch_size,
        args.max_engine_batch_size,
        calibration_stream,
        calibration_table,
    )


if __name__ == "__main__":
    main()
