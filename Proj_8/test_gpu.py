###Solely to test GPU on pytorch for engine.py

import torch
print("IS CUDA AVAILABLE? %r" % (torch.cuda.is_available()))

print("CUDA DEVICE COUNT? %s" % (torch.cuda.device_count()))

active_device = (torch.cuda.current_device())
print(torch.cuda.get_device_name(active_device))