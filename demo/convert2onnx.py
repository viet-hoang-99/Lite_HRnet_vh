import torch
import argparse
import torch.backends.cudnn as cudnn
import _init_paths
import models
from config import cfg
from config import update_config
import torch.onnx
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='experiments/coco/lite_hrnet/lite_hrnet_30_384x288.yaml')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def convert():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )
    pose_model.eval()
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    test_input=torch.randn(1,3,cfg.MODEL.IMAGE_SIZE[1],cfg.MODEL.IMAGE_SIZE[0]).to(CTX)
    print("dummy input test")
    torch.onnx.export(pose_model.module,test_input,"lite_hrnet.onnx",export_params=True,opset_version=11)
    print('Model has been converted to ONNX') 


if __name__=="__main__":
    convert()

