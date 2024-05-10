from models.multi_lenet import MultiLeNetO, MultiLeNetR
from models.segnet import SegnetEncoder, SegnetInstanceDecoder, SegnetSegmentationDecoder, SegnetDepthDecoder
from models.pspnet import SegmentationDecoder, get_segmentation_encoder
from models.multi_faces_resnet import ResNet, FaceAttributeDecoder, BasicBlock
import torchvision.models as model_collection
import torch.nn as nn
import torch

def get_model(params):
    data = params['dataset']
    if 'mnist' in data:
        model = {}
        model['rep'] = MultiLeNetR()
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].cuda()
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO()
            if params['parallel']:
                model['L'] = nn.DataParallel(model['L'])
            model['L'].cuda()
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO()
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['R'].cuda()
        return model

    if 'cityscapes' in data:
        model = {}
        model['rep'] = get_segmentation_encoder() # SegnetEncoder()
        #vgg16 = model_collection.vgg16(pretrained=True)
        #model['rep'].init_vgg16_params(vgg16)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].cuda()
        if 'S' in params['tasks']:
            model['S'] = SegmentationDecoder(num_class=19, task_type='C')
            if params['parallel']:
                model['S'] = nn.DataParallel(model['S'])
            model['S'].cuda()
        if 'I' in params['tasks']:
            model['I'] = SegmentationDecoder(num_class=2, task_type='R')
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['I'].cuda()
        if 'D' in params['tasks']:
            model['D'] = SegmentationDecoder(num_class=1, task_type='R')
            if params['parallel']:
                model['D'] = nn.DataParallel(model['D'])
            model['D'].cuda()
        return model

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder()
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].cuda()
        return model
    
    if 'sarcos' in data:
        model = {}
        model['rep'] = TwoLayerNet().cuda().double()
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        for t in params['tasks']:
            model[t] = Dummy(task=t)
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
            model[t].cuda()
        return model

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim = 21, hidden_dim = 1, output_dim = 3, bias = False):
        super(TwoLayerNet, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        # Second fully connected layer that outputs our 3 labels
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
    def forward(self, x, mask):
        o = self.fc1(x)
        o = self.fc2(o)
        return o, mask
    
class Dummy(nn.Module):
    # This is a dummy model that returns the input as output
    def __init__(self, task = 0):
        super(Dummy, self).__init__()
        self.task = int(task)

    def forward(self, x, mask):
        return x[:,self.task]+.01*torch.ones_like(x[:,self.task]), mask