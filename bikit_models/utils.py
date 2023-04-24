'''
Utils to make prediction with models
'''
import os
import glob
import sys
import time

from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

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
		process_img('assets/ExImg.jpg')
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

def preprocess_img(img):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.resize((224,224))
    img_np = np.array(img)

    img_np = (img_np / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_np = img_np.transpose(2, 0, 1)
    img = torch.from_numpy(img_np)
    img = img.unsqueeze_(0)
    img = img.type(torch.FloatTensor)
    return img


def _print_prediction_bar(prediction_probability, label):
    assert (prediction_probability>=0.0) and (prediction_probability<=1.0)
    bar_size = 40
    bar = '█' * int(bar_size * prediction_probability)
    bar = bar + '' * int(bar_size * (1-prediction_probability))

    sys.stdout.write(f"{label.ljust(20)} [{bar:{bar_size}s}] {prediction_probability*100:>6.2f}% \n")
    sys.stdout.flush()

def make_prediciton(model, img, cat_to_name, print_predictions=True, preprocess_image=False, activation=True):
	# Read image if it is a string
    if isinstance(img, str):
        img = Image.open(img_path)
    # Preprocess image
    if preprocess_image:
        img = preprocess_img(img)

    model.eval()
    tic = time.perf_counter()
    with torch.no_grad():
        logits = model(img)
    probabilities = torch.sigmoid(logits).numpy()[0] if activation else logits.numpy()[0]
    predictions = probabilities > 0.5
    toc = time.perf_counter()

    if print_predictions:
        n_classes = len(cat_to_name)
        for i in range(n_classes):
            label_name = cat_to_name[i]
            _print_prediction_bar(probabilities[i], label_name)
        print(f"Inference time (CPU): {(toc - tic)*1000:0.2f} ms")



if __name__ == '__main__':
	'''
	Quick test
	'''
	from daclnet import build_dacl
	img_path = "assets/image_0000468_crop_0000001.png"
	cp_path = "models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ResNet50_hta.pth"
	model, cat_to_name = build_dacl(cp_path=cp_path)
	img = process_img_daclnet(img_path)
	make_prediciton(model, img, cat_to_name)

	# from vistranet_0 import build_vistra_0
	# img_path = "assets/image_0000468_crop_0000001.png"
	# cp_path = "models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_0.pth"
	# model, cat_to_name = build_vistra_0(cp_path=cp_path)
	# img = process_img_vistranet(img_path)
	# make_prediciton(model, img, cat_to_name)

	# from vistranet_1 import build_vistra_1
	# img_path = "assets/image_0000468_crop_0000001.png"
	# cp_path = "models/checkpoints/codebrim-classif-balanced/codebrim-classif-balanced_ViT_s8_1.pth"
	# model, cat_to_name = build_vistra_1(cp_path=cp_path)
	# img = process_img_vistranet(img_path)
	# make_prediciton(model, img, cat_to_name, activation=False) # activation is already performed 


	# from models.vistranet_0 import build_vistra
	# from daclnet import build_dacl
	# img_path = 'assets\image_0000468_crop_0000001.png'
	# img_proc = process_img_vistranet(img_path)
    # # Instantiate the model:
	# model, cat_to_name = build_dacl(cp_path='models\checkpoints\meta4+dacl1k\meta4+dacl1k_MobileNetV3-Large_hta.pth')
	# model.eval()
	# with torch.no_grad(): # Disable tracking of gradients in autograd (saves some time)
	# 	logits = model(img_proc)
	# 	preds = torch.sigmoid(logits).float().squeeze(0) 
	
	# # Make a dict with the predictions:
	# preds_dict = {v:round(preds[int(k)].item(),4) for k,v in cat_to_name.items()}
	# print(preds_dict)
    # # View the classified image and it's predictions:
	# view_classify(img_path, preds_dict)

	# for cp_path in glob.glob('./models/checkpoints/codebrim-classif-balanced/*EfficientNetV1-B0_hta.pth'):
	# 	if cp_path.endswith(".pth"):	
	# 		print(cp_path)
	# 		cp_name = Path(cp_path).stem
	# 		model, cat_to_name = build_dacl(cp_path=cp_path)
	# 		model.eval()
	# 		example = torch.rand(1, 3, 224, 224)
	# 		traced_script_module = torch.jit.trace(model, example)	
	# 		traced_script_module.save(Path("models/jit_models") / (cp_name+'.pt'))
