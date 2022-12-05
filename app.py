# import argparse
import os

from PIL import Image

import numpy as np
import torch
import cv2
import sys

import utils.architecture.architecture as arch
import utils.imgops as ops

from flask import Flask, Response, request


app = Flask(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument('--input', default='input', help='Input folder')
# parser.add_argument('--output', default='output', help='Output folder')
# parser.add_argument('--reverse', help='Reverse Order', action='store_true')
# parser.add_argument('--tile_size', default=512,
#                     help='Tile size for splitting', type=int)
# parser.add_argument('--seamless', action='store_true',
#                     help='Seamless upscaling')
# parser.add_argument('--mirror', action='store_true',
#                     help='Mirrored seamless upscaling')
# parser.add_argument('--replicate', action='store_true',
#                     help='Replicate edge pixels for padding')
# parser.add_argument('--cpu', action='store_true',
#                     help='Use CPU instead of CUDA')
# parser.add_argument('--ishiiruka', action='store_true',
#                     help='Save textures in the format used in Ishiiruka Dolphin material map texture packs')
# parser.add_argument('--ishiiruka_texture_encoder', action='store_true',
#                     help='Save textures in the format used by Ishiiruka Dolphin\'s Texture Encoder tool')
# args = parser.parse_args()

# if not os.path.exists(args.input):
# 	print('Error: Folder [{:s}] does not exist.'.format(args.input))
# 	sys.exit(1)
# elif os.path.isfile(args.input):
# 	print('Error: Folder [{:s}] is a file.'.format(args.input))
# 	sys.exit(1)
# elif os.path.isfile(args.output):
# 	print('Error: Folder [{:s}] is a file.'.format(args.output))
# 	sys.exit(1)
# elif not os.path.exists(args.output):
# 	os.mkdir(args.output)

device = 'cuda'

ishiiruka_texture_encoder = False
ishiiruka = False
tile_size = 512

NORMAL_MAP_MODEL = 'utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = 'utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'


def process(img, model):
	img = img * 1. / np.iinfo(img.dtype).max
	img = img[:, :, [2, 1, 0]]
	img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
	img_LR = img.unsqueeze(0)
	img_LR = img_LR.to(device)

	output = model(img_LR).data.squeeze(
		0).float().cpu().clamp_(0, 1).numpy()
	output = output[[2, 1, 0], :, :]
	output = np.transpose(output, (1, 2, 0))
	output = (output * 255.).round()
	return output

def load_model(model_path):
	global device
	state_dict = torch.load(model_path)
	model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
							mode='CNA', res_scale=1, upsample_mode='upconv')
	model.load_state_dict(state_dict, strict=True)
	del state_dict
	model.eval()
	for k, v in model.named_parameters():
		v.requires_grad = False
	return model.to(device)


@app.route('/generate', methods=['OPTIONS', 'POST'])
def generate_maps():
	if (request.method == 'OPTIONS'):
		print('got options 1')
		response = Response()
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		print('got options 2')
		return response

	body = request.get_data()
	img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

	models = [
		# NORMAL MAP
		load_model(NORMAL_MAP_MODEL), 
		# ROUGHNESS/DISPLACEMENT MAPS
		load_model(OTHER_MAP_MODEL)
	]

	# modes: seamless, mirror, replicate
	seamless_mode = request.args.get('mode')
	# maps: n (for normal map), r (for roughness), d (for displacement)
	image_map = request.args.get('map')

	# Seamless modes
	if seamless_mode == 'seamless':
		img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
	elif seamless_mode == 'mirror':
		img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
	elif seamless_mode == 'replicate':
		img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

	img_height, img_width = img.shape[:2]

	# Whether or not to perform the split/merge action
	do_split = img_height > tile_size or img_width > tile_size

	if do_split:
		rlts = ops.esrgan_launcher_split_merge(img, process, models, scale_factor=1, tile_size=tile_size)
	else:
		rlts = [process(img, model) for model in models]

	if seamless_mode == 'seamless' or seamless_mode == 'mirror' or seamless_mode == 'replicate':
		rlts = [ops.crop_seamless(rlt) for rlt in rlts]

	normal_map = rlts[0]
	roughness = rlts[1][:, :, 1]
	displacement = rlts[1][:, :, 0]

	if ishiiruka_texture_encoder:
		r = 255 - roughness
		g = normal_map[:, :, 1]
		b = displacement
		a = normal_map[:, :, 2]
		output = cv2.merge((b, g, r, a))
		# cv2.imwrite(os.path.join(output_folder, '{:s}.mat.png'.format(base)), output)
	else:
		# cv2.imwrite(os.path.join(output_folder, normal_name), normal_map)
		rough_img = 255 - roughness if ishiiruka else roughness
		# cv2.imwrite(os.path.join(output_folder, rough_name), rough_img)
		# cv2.imwrite(os.path.join(output_folder, displ_name), displacement)

		if image_map == 'n':
			result = cv2.imencode('.png', normal_map)[1].tobytes()
		elif image_map == 'r':
			result = cv2.imencode('.png', rough_img)[1].tobytes()
		elif image_map == 'd':
			result = cv2.imencode('.png', displacement)[1].tobytes()
		response = Response(result, headers={'Content-Type':'image/png'})
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Headers'] = '*'
		response.headers['Access-Control-Allow-Methods'] = '*'
		response.headers['Access-Control-Expose-Headers'] = '*'
		response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
		response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
		response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
		return response


if __name__ == '__main__':
	app.run(
		host='0.0.0.0',
		port=8080,
		threaded=True,
		debug=False
	)
