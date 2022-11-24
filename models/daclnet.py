from torch import nn
from torchvision import models

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

# Dictionary to find the suiting EfficientNet model according to the resolution of the input-images:
efnet_dict = {'b0': 224, 'b1': 240, 'b2': 260, 'b3': 300,   
              'b4': 380, 'b5': 456, 'b6': 528, 'b7': 600    
             }

class DaclNet(nn.Module):
    def __init__(self, base_name, resolution, hidden_layers, num_class, drop_prob=0.2, freeze_base=True):
        ''' 
        Builds a network separated into a base model and classifier with arbitrary hidden layers.
        
        Attributes
        ---------
        base_name:      string, basemodel for the NN
        resolution:     resolution of the input-images, example: 224, 240...(look efnet_dic), Only needed for EfficientNet
        hidden_layers:  list of integers, the sizes of the hidden layers
        drop_prob:      float, dropout probability
        freeze_base:    boolean, choose if you want to freeze the parameters of the base model
        num_class:      integer, size of the output layer according to the number of classes

        Example
        ---------
        model = Network(base_name='efficientnet', resolution=224, hidden_layers=[32,16], num_class=6, drop_prob=0.2, freeze_base=True)
        
        Note
        ---------
        -print(efficientnet) -> Last module: (_swish): MemoryEfficientSwish() and the last fc-layers are displayed
         This activation won't be called during forward due to: "self.base.extract_features"! No activation of last layer!
        '''
        super(DaclNet, self).__init__()
        # basemodel
        self.base_name = base_name
        self.resolution = resolution
        self.hidden_layers = hidden_layers
        self.freeze_base = freeze_base

        if self.base_name == 'mobilenet':
            base = models.mobilenet_v3_large(pretrained=True) 
            modules = list(base.children())[:-1] 
            self.base = nn.Sequential(*modules)
            # for pytorch model:
            if hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[0].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[0].in_features, num_class)

            self.activation = nn.Hardswish()

        elif self.base_name == 'resnet':
            base = models.resnet50(pretrained=True) 
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(base.fc.in_features, num_class)   
            self.activation = nn.ELU() 

        elif self.base_name == 'efficientnet':      
            for ver in efnet_dict:
                if efnet_dict[ver] == self.resolution:
                    self.version = ver
                    full_name = self.base_name+'-'+ver
            self.base = EfficientNet.from_pretrained(model_name=full_name) 
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(self.base._fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(self.base._fc.in_features, num_class)   
            self.activation = MemoryEfficientSwish()
            
        elif self.base_name == 'mobilenetv2':
            base = models.mobilenet.mobilenet_v2(pretrained=True)
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[1].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[1].in_features, num_class)
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError    
        
        # freeze the base
        if self.freeze_base:
            for param in self.base.parameters(): 
                param.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=drop_prob, inplace=True)

        # classifier
        # Add a variable number of more hidden layers
        if self.hidden_layers:
            layer_sizes = zip(self.hidden_layers[:-1], self.hidden_layers[1:])        
            self.classifier.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            # Add output layer to classifier
            self.classifier.append(nn.Linear(self.hidden_layers[-1], num_class))
        else:
            pass
        
    def forward(self, input_batch):
        ''' 
        Performs the feed-forward process for the input batch and returns the logits

        Arguments
        ---------
        input_batch: torch.Tensor, Multidimensional array holding elements of datatype: torch.float32, 
                     it's shape is: [1, 3, 224, 224] according to N x C x H x W,
                     The input batch carries all pixel values from the images inside teh batch
        Note
        ---------
        Every model uses 2d-Average-Pooling with output_size=1 after the feature extraction or rather before flattening.
        The pooling layer of ResNet50 and MobileNetV3 was kept in the squential -> Doesn't have to be called in forward!
        EffNet had to be implemented with the AdaptiveAvgpool2d in this forward function because of missing pooling when
        calling: "effnet.extract_features(input_batch)"
        Also mobilenetV2 needs the manually added pooling layer.

        Returns
        ---------
        logits: torch.Tensor, shape: [1, num_class], datatype of elements: float
        '''
        # Check if model is one that needs Pooling layer and/or special feature extraction
        if self.base_name in ['efficientnet', 'mobilenetv2']:
            if self.base_name == 'efficientnet':
                x = self.base.extract_features(input_batch)
            else:
                # For MobileNetV2
                x= self.base(input_batch)
            pool = nn.AdaptiveAvgPool2d(1)
            x = pool(x)
        else:
            # For any other model don't additionally apply pooling:
            x = self.base(input_batch)
        
        x = self.dropout(x)         # Originally only in EfficientNet a Dropout after feature extraction is added  
        x = x.view(x.size(0), -1)   # Or: x.flatten(start_dim=1)
        if self.hidden_layers:    
            for i,each in enumerate(self.classifier):
                # Put an activation function and dropout after each hidden layer
                if i < len(self.classifier)-1:
                    x = self.activation(each(x))
                    x = self.dropout(x)
                else:
                    # Don't use an activation and dropout for the last layer
                    logits = each(x)
                    break
        else:
            logits = self.classifier(x)

        return logits