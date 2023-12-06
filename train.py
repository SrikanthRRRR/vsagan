"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
import torch
import gc, time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

##
def main():
    """ Training
    """
    opt = Options().parse()
    if opt.clnmem == 1:
        for _ in range(1,3,1):
            cleanUpMemory()

    data = load_data(opt)
    
    model = load_model(opt, data)

    pytorch_total_params_g = sum(p.numel() for p in model.netg.parameters() if p.requires_grad)

    pytorch_total_params_d = sum(p.numel() for p in model.netd.parameters() if p.requires_grad)

    model.train()
    print("Training completed for dataset:", opt.dataset, ", with batch size:", opt.batchsize, ", with epochs:", opt.niter, ", with lr decay:", opt.niter_decay, ", with beta:", opt.beta1, ", with lr:", opt.lr
          , ", -" , model.best_auc , "-", model.best_auc_epoch, "\nwith loss weight(adv)", opt.w_adv, ", with loss weight (con)", opt.w_con, ", with loss weight (lat)", opt.w_lat, ", with loss weight (var)", opt.w_var
          , ", with CBAM", opt.addCBAM, ", with g_lat_dim:", opt.g_lat_dim, ", with DSC:", opt.DSC, ", with channels:",opt.nc)
    cleanUpMemory()

def cleanUpMemory():
    gc.collect()
    torch.cuda.empty_cache()
    #del variables
    min_memory_available = 4 * 1024 * 1024 * 1024  # 2GB
    wait_until_enough_gpu_memory(min_memory_available)

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    print(torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("device", str(i))
        handle = nvmlDeviceGetHandleByIndex(i)

        for _ in range(max_retries):
            info = nvmlDeviceGetMemoryInfo(handle)
            print(info.free)
            if info.free >= min_memory_available:
                break
            print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")


if __name__ == '__main__':
    main()
