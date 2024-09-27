import os
import cv2
import glob
import onnx
import torch
import warnings
import argparse
import onnxruntime
import numpy as np
from tqdm import tqdm
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from ImageFormatReader import ImageFormatReader
from wsi_test_dataloader import testGenerator_ttpc


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


def parse_args():
    parser = argparse.ArgumentParser(description="Infer with TensorRT engine.")

    parser.add_argument("--batch_size", type=int, help="data batch size", default=1)
    parser.add_argument("--padding", type=int, help="padding of data", default=512)

    parser.add_argument("--output_shape", help="output shape", default=(5, 2048, 2048))
    parser.add_argument("--tile_size", help="tile size", default=2048, type=int)

    parser.add_argument(
        "--wsi_names",
        help="sample image folder path with .svs files",
        default=[
            "/mnt/prj003/projetcs/artifacts/data/TCGA_mix/WSI/69a0b7cf-1e0b-4db9-96a4-6395abfc2393/TCGA-2H-A9GH-01Z-00-DX1.B2BF80D6-D348-4C5F-A205-6827684BF3B6.svs",
        ],
    )

    parser.add_argument(
        "--tensorrt_engine_path",
        help="tensorrt engine path",
        default="int8_implicit.engine",
    )

    parser.add_argument(
        "--prediction_path",
        type=str,
        help="path predictions",
        default="/mnt/prj003/users/tanya/quant_pred/",
    )

    args = parser.parse_args()

    return args


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, output_shape, max_batch_size=1, dtype=np.float32):
        self.output_shape = output_shape
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=1):
        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host.reshape(self.output_shape) for out in self.outputs]


def predict_seg(
    fHandle, mask, model_input_size, padding, batch_size, model_name, image_shape
):
    tumor_mask = np.zeros(mask.shape)
    ValidDataset = testGenerator_ttpc(
        fHandle,
        mask,
        model_input_size=model_input_size,
        stride=padding,
        mask_at=2.5,
        required_mag=2.5,
    )
    ValidDataLoader = torch.utils.data.DataLoader(
        ValidDataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    effective_tile_size: int = model_input_size - 2 * padding

    input_names = ["TumorInput"]
    output_names = ["TumorOutput"]
    EP_list = ["CUDAExecutionProvider"]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    inst_info_dict = []
    time_n = []
    inst_iter = 0
    for x, patch_name in tqdm(ValidDataLoader):
        test_image = x.numpy()

        model = TrtModel(model_name, image_shape)
        shape = model.engine.get_binding_shape(0)

        starter.record()

        outputs = model(test_image, batch_size)

        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        time_n.append(curr_time)

        print(f"Time took per patch {np.round(curr_time, 3)} seconds")
        outputs = np.array(outputs)

        seg_pred = np.argmax((outputs), axis=1)

        for n in range(len(seg_pred)):
            tumor = seg_pred[n, :, :]
            x_pos = int(patch_name[n].split(".")[0].split("_")[1])
            y_pos = int(patch_name[n].split(".")[0].split("_")[0])
            tumor = tumor[
                padding : effective_tile_size + padding,
                padding : effective_tile_size + padding,
            ]
            try:
                tumor_mask[
                    y_pos * effective_tile_size : (y_pos + 1) * effective_tile_size,
                    x_pos * effective_tile_size : (x_pos + 1) * effective_tile_size,
                ] = tumor
            except:
                try:
                    size = tumor_mask.shape[0] - y_pos * effective_tile_size
                    tumor_mask[
                        y_pos * effective_tile_size :,
                        x_pos * effective_tile_size : (x_pos + 1) * effective_tile_size,
                    ] = tumor[0:size, :]
                except:
                    try:
                        size = tumor_mask.shape[1] - x_pos * effective_tile_size
                        tumor_mask[
                            y_pos
                            * effective_tile_size : (y_pos + 1)
                            * effective_tile_size :,
                            x_pos * effective_tile_size :,
                        ] = tumor[:, 0:size]
                    except:
                        sizex = tumor_mask.shape[1] - x_pos * effective_tile_size
                        sizey = tumor_mask.shape[0] - y_pos * effective_tile_size
                        tumor_mask[
                            y_pos * effective_tile_size :,
                            x_pos * effective_tile_size :,
                        ] = tumor[0:sizey, 0:sizex]

    temp_pred = np.zeros((tumor_mask.shape[0], tumor_mask.shape[1], 3), np.uint8)
    temp_pred[tumor_mask == 1.0] = (0, 0, 255)
    temp_pred[tumor_mask == 2.0] = (255, 0, 0)
    temp_pred[tumor_mask == 3.0] = (0, 255, 0)
    temp_pred[tumor_mask == 4.0] = (255, 255, 0)
    print("Final time per WSI", np.sum((time_n[1:])))
    return temp_pred, np.sum((time_n[1:]))


def find_name(path):
    name = os.path.basename(path)
    name_part = name.split(".")
    if name_part[-1] == "svs":
        return_name = ""
        for part in name_part[:-2]:
            return_name = part + "."
        return_name = return_name[:-2]
        return return_name
    else:
        name_part = name.split("_")
        return_name = ""
        for part in name_part:
            return_name += part + "_"
        return_name = return_name[:-1]
        return_name = return_name[:-5]
        return return_name


def main():
    args = parse_args()
    all_wsi_paths = {}
    avg_time = []

    for path in args.wsi_names:
        return_name = find_name(path)
        name = os.path.basename(path)
        name_part = name.split(".")
        if name_part[-1] == "svs":
            all_wsi_paths[return_name] = path
        else:
            return_name = return_name.split(".")[0]
            all_wsi_paths[return_name] = path

    for wsi_path in tqdm(args.wsi_names):
        name = ""
        part_name = os.path.basename(wsi_path).split("_")[0:-1]
        for part in part_name:
            name = name + part + "_"
        name = name[:-1]
        fHandle = ImageFormatReader(wsi_path, args.tile_size, 2)
        ifd = fHandle.getIFDFromResolution(1, 2.5)
        thumb_nail = fHandle.getFullImageDataFromIFD(ifd)
        thumb_nail = cv2.cvtColor(thumb_nail, cv2.COLOR_RGB2BGR)
        wsi_name = find_name(wsi_path)

        prediction, time_taken = predict_seg(
            fHandle,
            thumb_nail[:, :, 0],
            args.tile_size,
            args.padding,
            args.batch_size,
            args.tensorrt_engine_path,
            args.output_shape,
        )
        avg_time.append(time_taken)

        cv2.imwrite(args.prediction_path + wsi_name + "1_mask.png", prediction)
        cv2.imwrite(args.prediction_path + wsi_name + "_img.png", thumb_nail)

    print("Average time per WSI", np.mean(avg_time))


if __name__ == "__main__":
    main()
