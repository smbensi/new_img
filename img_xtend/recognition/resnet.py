from urllib.parse import urlsplit

import numpy as np
import cv2
import torch
import tritonclient
from img_xtend.utils.triton import TritonRemoteModel 

USE_TRITON = True   #TODO add  to the config

class ResNetModel: 
    def __init__(
        self,
        similarity_tresh=0.75,
        new_person_thresh=1.25,
        ):
        
        self.similarity_thresh = similarity_tresh
        self.new_person_thresh = new_person_thresh
        
        self.recognized_faces = []
        self.unrecognized_faces = []
        
        if USE_TRITON:
            self.path = 'http://localhost:8000/resnet'
            try:
                self.model = TritonRemoteModel(self.path)
            except ConnectionRefusedError as e:
                print(f"ERROR: {e} \nCheck that the Triton server is correctly loaded")
                exit()
            except tritonclient.utils.InferenceServerException as e:
                print(f"ERROR: {e} \nCheck that the model is correctly loaded in the Triton Server")
                exit()
        else:
            self.load_local_model()

    def __call__(self, inputs:np.ndarray):
        """
        Call the model with the given inputs 
        """
        outputs = []
        for x in [inputs]:
            if USE_TRITON:
                output = self.model(self.transform(x))
                outputs.append(output[0])
            else:
                output = self.get_embedding_local(self.transform(x))
                outputs.append(output)
        # FIXME  consistent output  numpy array / torch tensor
        return outputs

    def transform(self, input:np.ndarray):
        input = cv2.resize(input, (160,160)) # to be adapted to our Resnet
        input = np.transpose(input, (2,0,1))
        input = np.expand_dims(input, axis=0).astype(np.float32)
        return input
    
    def load_local_model(self):  # TODO check that these 2 functions work 
        """If the model is not loaded on the triton server we are loading it manually
        be careful to use a docker that has access to cuda and tensorRT (much heavier in memory)"""
        # TODO add a check that we have access to TensorRT and CUDA
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        TRT_LOGGER = trt.Logger()
        engine_file = cfg["local_trt_resnet_model"]  # TODO  add the cfg
        runtime = trt.Runtime(TRT_LOGGER)
        f = open(engine_file, 'rb')
        self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def get_embedding_local(self,face_resized):
        """Run inference of TensorRT model on GPU loaded manually """
        input_img = face_resized
        bindings = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(input_img)
                    input_memory = cuda.mem_alloc(input_img.nbytes)
                    bindings.append(int(input_memory))
            else:
                output_size = (512)
                output_buffer = cuda.pagelocked_empty(output_size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
            
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        self.context.execute_async_v2(
            bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        embedding = torch.from_numpy(output_buffer)
        return embedding
    
    def update(self, img):
        pass
    


# test
if __name__=="__main__":
    a = ResNetModel()

    img = cv2.imread("Jake/2022-01-23-180541_1.jpg")
    print(img.dtype)
    print(img.shape)
    output = a(img)
    print(f'{output[0].shape = }')