from .Base import *


class SoftMax(BaseLayer):
    def forward(self, input_tensor):
        exp = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.input_tensor = exp / np.sum(exp, axis=1, keepdims=True)
        return self.input_tensor

    def backward(self, error_tensor):   
        
        # derivatives = np.zeros((self.input_tensor.shape[0], self.input_tensor.shape[1], self.input_tensor.shape[1]))
        # for b in range(self.input_tensor.shape[0]):
        #     for i in range(self.input_tensor.shape[1]):
        #         for j in range(self.input_tensor.shape[1]):
        #             if i == j:
        #                 derivatives[b, i, j] = self.input_tensor[b, i] * (1- self.input_tensor[b, i])
        #             else:
        #                 derivatives[b, i, j] = -self.input_tensor[b, i] * self.input_tensor[b, j]
        
        # tensor1 = np.einsum('ij,ik->ijk', self.input_tensor, self.input_tensor)
        # tensor2 = np.einsum('ij,jk->ijk', self.input_tensor, np.eye(self.input_tensor.shape[1], self.input_tensor.shape[1]))
        # derivatives = tensor2 - tensor1

        # new_error_tensor = np.zeros((derivatives.shape[0], derivatives.shape[1]))
        # for i in range(derivatives.shape[0]):
        #     new_error_tensor[i] = error_tensor[i] @ derivatives[i] 
        # return new_error_tensor

        # return np.einsum('bi,bij->bj', error_tensor, derivatives)

        return self.input_tensor * (error_tensor - np.sum(error_tensor * self.input_tensor, axis=1, keepdims=True))