'''
Utils to make prediction with models
'''
import os

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import transforms

# Preprocessing-functions:
def process_img_daclnet(img_path=None):
	''' 
	Scales, crops, and normalizes a PIL image for a PyTorch model,
	returns a Torch Tensor
	Args: 
		img_path: 	string, filepath of the image
	Example: 
		process_img('test/1/image_06743.jpg')
	Returns: 
		torch.float32 of shape: [1, 3, 224, 224]
	'''

	if not img_path:
		print('Parse the filename of the image!')
	else:
		#Parse image as PIL Image
		image = Image.open(img_path)
		# Setting Resize Parameters (width and height)
		image_ratio = image.height / image.width
		if  image.width < image.height  or image.width > image.height:
			if image.width < image.height:
				resize = (256, int(image_ratio * 256))
			else:
				resize = (int(256 / image_ratio), 256)
		else:
			resize = (256, 256)
		
		#Setting Crop parameters
		crop_size = 224
		crop_x = int((resize[0] - crop_size) / 2)
		crop_y = int((resize[1] - crop_size) / 2)
		crop_box = (crop_x, crop_y,crop_x + crop_size, crop_y+crop_size)
	  	
		#Transformation
		pil_image = image.resize(resize)
		pil_image = pil_image.crop(crop_box)
		np_image = np.array(pil_image)
		np_image = (np_image/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
		np_image = np_image.transpose(2,0,1)
		image = torch.from_numpy(np_image)
		image = image.unsqueeze_(0)
		image = image.type(torch.FloatTensor)
		return image

def process_img_vistranet(img_path):
	''' 
	Function for preprocessing images according to Sofia who submitted via https://dacl.ai/.
	All from: https://github.com/mpaques269546/codebrim_challenge.
  	Args:
  		img_path:	string, path to image you want to classify. 
  		show_img:	display image
  	Returns:
  		img: 		image as torch.Tensor of shape: [1, 3, 224, 224]
  	'''
	transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
	if os.path.isfile(img_path):
		with open(img_path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')
	img = transform(img)
	return img.unsqueeze(0)

def view_classify(img_path, result_dict):
	''' 
	Function for viewing an image, its predicted classes and the probabilities
  	in a horizontal bar chart.
  	Args:
  		image_path:		string, path to image you want to classify. You can take random_image_path 
		  				function so a random image from test-folder will be classified
  		result_dict:	dict, result_dict returned by the predict function
  	Returns:
  		None - just displays the image next to the bar chart  
  	'''
	result_list = list(result_dict.items())
	result_list = sorted(result_list, reverse=False, key=lambda result: result[1])
	cat_names = [x[0] for x in result_list]
	ps = [x[1] for x in result_list]	
	fig, (ax1, ax2) = plt.subplots(figsize=(9,12), ncols=2)
	ax1.imshow(plt.imread(img_path))
	ax1.axis('off')
	
	# create title:
	title = result_list[-1][0]
	for i in range((len(result_list)-2), 0, -1):
		if result_list[i][1] > .5:
			title += (', ' + result_list[i][0])
	ax1.set_title(title)

	ax2.barh(range(len(cat_names)), ps, align='center')
	ax2.set_aspect(0.1)
	ax2.set_yticks(np.arange(len(cat_names)))
	ax2.set_yticklabels(cat_names, size='small')
	ax2.set_title('Class Probability')
	ax2.set_xlim(0, 1.1)

	plt.tight_layout()


if __name__ == '__main__':
	'''
	Quick test
	'''
	from vistranet import build_model
	from daclnet import DaclNet

	img_path = 'assets\image_0000761_crop_0000006.png'
	img_proc = process_img_vistranet(img_path)

    # Instantiate the model:
	model = build_model(pretrained_weights='models\checkpoints\codebrim-classif-balanced\codebrim-classif-balanced_ViT_s8.pth', img_size=224, num_cls=6, quantized=True)
    # Load the checkpoint:
    
    # Choose which checkpoint/model you want to load from the table above:
	cp_name = 'models\checkpoints\codebrim-classif-balanced\codebrim-classif-balanced_ResNet50_hta.pth'
	cp = torch.load(cp_name)
    # model = DaclNet(base_name=cp['base'], resolution = cp['resolution'], hidden_layers=cp['hidden_layers'], 
	# 			num_class=cp['num_class'])
    # model.load_state_dict(cp['state_dict']) # Load the pre-trained weights into the model
	
	model.eval()
	with torch.no_grad(): # Disable tracking of gradients in autograd (saves some time)
		preds = model(img_proc)
		# preds = torch.sigmoid(logits).float().squeeze(0) # Nur bei DaclNet notwendig
	
	# Make a dict with the predictions:
	cat_to_name = {0:'NoDamage', 1: 'Crack', 2:'Spalling', 3:'Efflorescence', 4:'BarsExposed', 5:'Rust'}
	preds = preds.flatten()
	preds_dict = {v:round(preds[int(k)].item(),3) for k,v in cat_to_name.items()}
	print(preds_dict)
    # View the classified image and it's predictions:
	# view_classify(img_path, preds_dict)
	# logits = logits.flatten()
    # Apply sigmoid activation to get predictions:
    # preds = torch.sigmoid(logits).float().squeeze(0).to('cpu')
    
    # labels_list =  ['NoDamage' , 'Crack', 'Spalling', 'Efflorescence', 'BarsExposed', 'Rust']
    # make_predictions(model, "../assets/ExImg.jpg", labels= labels_list)