import os
import torch 
import torch.nn as nn

# from cam_model import cam_resnet
import resnet

from collections import OrderedDict



class Config:
    def __init__(self):
        self.sample_size = 64
        self.sample_duration = 16
        self.shortcut_type = 'B'
        self.num_classes = 8
        self.ckpt = './models/Fold0_best_two_models.pth.tar'


class ResNetConfig:
    def __init__(self):
        self.sample_size = 48
        self.sample_duration = 32
        self.model_depth = 50
        self.shortcut_type = 'A'
        self.use_dropout = False
        self.activation = 'relu'
        self.att_block = 'none'  
        self.cov = False  
        
        self.ckpt_path = './ckpts/lidc_res50_0931.pth.tar'


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.model = cam_resnet(
            sample_size=self.config.sample_size,
            sample_duration=self.config.sample_duration,
            shortcut_type=self.config.shortcut_type,
            num_classes=self.config.num_classes
        )

        self._load_ckpt()

    def _load_ckpt(self):
        if not os.path.exists(self.config.ckpt):
            raise Exception('Checkpoint file not exists!')
        ckpt = torch.load(self.config.ckpt)['state_dict']
        try:
            self.model.load_state_dict(ckpt)
        except:
            import pdb; pdb.set_trace()


    def forward(self, x):
        return self.model(x)

def _load_ckpt(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise Exception('Checkpoint file not exists!')
    ckpt = torch.load(ckpt_path)['state_dict']
    try:
        model.load_state_dict(ckpt)
    except:
        import pdb; pdb.set_trace()
    return model

    


def generate_attmodel(config):
    try:
        Model = getattr(resnet, 'resnet'+str(config.model_depth))
    except:
        print(config.model_depth)
        import pdb; pdb.set_trace()

    model = Model(
        sample_size=config.sample_size,
        sample_duration=config.sample_duration, 
        #use_dropout=config.use_dropout, 
        shortcut_type=config.shortcut_type, 
        #activation=config.activation,
        num_classes=config.num_classes
    )

    print('********** Load parameters ************')
    def _load_resnet_ckpt(model, ckpt_path):
        # import pdb; pdb.set_trace()
        if not os.path.exists(ckpt_path):
            raise Exception('Checkpoint file not exists!')
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        curmodel_dict = model.state_dict()
        # import pdb; pdb.set_trace()

        new_state_dict = OrderedDict()
        for k, v in ckpt.items():
            # import pdb; pdb.set_trace()
            # name = k[6:] # remove 'module.'
            name = k[6:] # remove 'model.'
            new_state_dict[name] = v
        new_state_dict = {k: v for k, v in new_state_dict.items()}
        if len(new_state_dict) == 0:
            import pdb; pdb.set_trace()
        curmodel_dict.update(new_state_dict)

        try:
            model.load_state_dict(curmodel_dict)
        except:
            import pdb; pdb.set_trace()
        return model
    
    def _load_ckpt_v2(model, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise Exception('Checkpoint file not exists!')
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        try:
            model.load_state_dict(ckpt)
        except:
            import pdb; pdb.set_trace()
        return model 
    
    model = _load_resnet_ckpt(model, config.ckpt_path)
    # model = _load_ckpt_v2(model, config.ckpt_path)
    return model



def generate_model(config):
    # config = Config()
    model = resnet(
        sample_size=config.sample_size,
        sample_duration=config.sample_duration,
        shortcut_type=config.shortcut_type,
        num_classes=config.num_classes)
    model = _load_ckpt(model, config.ckpt)
    return model 

if __name__ == '__main__':
    # model = generate_model()
    # import pdb; pdb.set_trace()
    # # model = Model(Config())
    # print(model)
    config = ResNetConfig()
    model = generate_attmodel(config)
    import pdb; pdb.set_trace()
    print(model)





