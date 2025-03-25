from resnest_gate_fusion import MultiScaleGatedSE_ResNeSt101e 
from resnest_pyramid_fusion import MultiScaleResNeStSE
import timm

# load the model which we wanna calculate
#model = timm.create_model('resnest101e', pretrained=True)
model = MultiScaleGatedSE_ResNeSt101e(num_classes=100)
#model = MultiScaleResNeStSE(num_classes=100)

# calculate the model parameters
num_params_all = sum(p.numel() for p in model.parameters())
print("Total parameters (all):", num_params_all)
