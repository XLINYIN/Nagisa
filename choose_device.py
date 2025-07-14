import pynvml
import torch

Choose_ID = -1   #   默认设备
MemFree = -1     #   随缘显存
Choose_Device = ''
if torch.cuda.is_available() :
   pynvml.nvmlInit()
   for i in range(torch.cuda.device_count()):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
      p = torch.cuda.get_device_properties(i)
      if meminfo.free >= MemFree:
           Choose_ID = i
           MemFree = meminfo.free
   Choose_Device = f'cuda:{Choose_ID}'
else :
    Choose_Device = 'cpu'