import torch
import transformers
import math
import torch.nn.functional as F
from utils import recursive_getattr, recursive_setattr


class LoRALinear(torch.nn.Module):
    def __init__(self, weight, bias, lora_dim, lora_scaling):
        super(LoRALinear, self).__init__()
        # Save original weight and bias
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        
        # 没有要求实现dropout
        # self.lora_dropout = torch.nn.Dropout(p=lora_dropout)
        
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        # # Actual trainable parameters
        # if r > 0:
        #     self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        #     self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        #     self.scaling = self.lora_alpha / self.r
        #     # Freezing the pre-trained weight matrix
        #     self.weight.requires_grad = False
        
        # TODO: Implement lora left and right weights
        self.lora_right_weight = torch.nn.Parameter(weight.new_zeros((lora_dim, weight.size(0))))
        self.lora_left_weight = torch.nn.Parameter(weight.new_zeros((weight.size(1), lora_dim)))
        #############################################
        self.lora_scaling = lora_scaling / lora_dim
        self.init_parameters()
        # TODO: Freeze original weight and bias
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        #######################################

    def init_parameters(self):
        # TODO: Initialize LoRA parameters
        torch.nn.init.kaiming_uniform_(self.lora_left_weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_right_weight)
        ##################################

    def forward(self, input):
        # TODO: Implement the forward function
        # print(input.shape) 
        # print(self.weight.shape) 
        # print(self.lora_left_weight.shape) 
        # print(self.lora_right_weight.shape)
        result = F.linear(input, self.weight, bias=self.bias)    
        # result += (self.lora_dropout(x) @ self.lora_left_weight @ self.lora_right_weight) * self.lora_scaling
        result += (input @ self.lora_left_weight @ self.lora_right_weight) * self.lora_scaling
        return result
        ######################################


def convert_linear_layer_to_lora(model, part_module_name, lora_dim=0, lora_scaling=1):
    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, transformers.pytorch_utils.Conv1D)) and part_module_name in name:
            replace_name.append(name)
    for name in replace_name:
        module = recursive_getattr(model, name)
        if isinstance(module, torch.nn.Linear):
            tmp = LoRALinear(module.weight, module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            tmp = LoRALinear(module.weight.t().detach(), module.bias, lora_dim, lora_scaling).to(module.weight.device).to(module.weight.dtype)
        else:
            raise ValueError("Unsupported module type")
        recursive_setattr(model, name, tmp)
    return model

# https://github.com/microsoft/LoRA/blob/main/loralib/utils.py
def only_optimize_lora_parameters(model):
    # TODO: Turn off the gradient of all the parameters except the LoRA parameters
    # Iterate through all modules in the model
    for name, param in model.named_parameters():
        # If the parameter is part of the LoRA parameters, keep it trainable
        if 'lora_' not in name:
            param.requires_grad = False
    return model
    ##############################################################################

def get_lora_state_dict(model):
    # TODO: return lora left and right weights as state dict
    # The saved state dict will be used later for loading
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    ########################################################