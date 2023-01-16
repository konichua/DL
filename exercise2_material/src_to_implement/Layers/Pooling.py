from .Base import *

class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape) -> None:
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = np.array(pooling_shape)
    
    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        output = np.zeros((input_tensor.shape[0], 
                           input_tensor.shape[1],
                           *(np.array(input_tensor.shape[2:]) - self.pooling_shape + 1)))
        max_pos = np.zeros((*output.shape, 2), dtype=int)
        for b in range(output.shape[0]):
            for c in range(output.shape[1]):
                for h in range(output.shape[2]):
                    for w in range(output.shape[3]):
                        current = input_tensor[b, c, 
                                               h:h + self.pooling_shape[0],             
                                               w:w + self.pooling_shape[1]]
                        output[b, c, h, w] = np.max(current)    
                        max_pos[b, c, h, w] = [h, w] + np.array(np.unravel_index(current.argmax(), current.shape))
        self.max_pos = max_pos[:, :, ::self.stride_shape[0], ::self.stride_shape[1], :] 
        return output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]    
        
    def backward(self, error_tensor):
        output = np.zeros(self.input_tensor_shape)
        for b in range(error_tensor.shape[0]):
            for c in range(error_tensor.shape[1]):
                for h in range(error_tensor.shape[2]):
                    for w in range(error_tensor.shape[3]):
                        # h_in_input = self.max_pos[b,c,h,w][0]
                        # w_in_input = self.max_pos[b,c,h,w][1]
                        output[b, c, self.max_pos[b,c,h,w][0] ,self.max_pos[b,c,h,w][1]] += error_tensor[b, c, h, w]
        return output