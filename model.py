'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn

def generate_model(opt):
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':   
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)


    if opt.device != 'cpu':
        model = model.to(opt.device)
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    return model, model.parameters()
