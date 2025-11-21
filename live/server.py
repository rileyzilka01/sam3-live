import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import zmq
import json

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)

processor = Sam3Processor(model, confidence_threshold=0.5)

def main():
	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind("tcp://*:4444")
	print("SAM3 segmentation server ready!")

	show_mask = False
	while True:
		parts = socket.recv_multipart()

		# ping
		try:
			message = json.loads(parts[0].decode("utf-8"))
			if message.get("ping") is True:
				socket.send_json({"pong": True})
				continue
		except json.JSONDecodeError:
			pass

		# check valid image message
		if len(parts) != 3:
			socket.send_json({"error": "invalid message"})
			continue

		# Unpack and save image
		header_bytes, img_bytes, prompt_bytes = parts
		header = json.loads(header_bytes.decode("utf-8"))
		h, w, c = header["height"], header["width"], header["channels"]

		img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((h, w, c))
		print(f"Received image: {img.shape}")

		image = Image.fromarray(img)

		prompt_msg = json.loads(prompt_bytes.decode("utf-8"))
		prompts = ["black end effector on robot arm", "cube", "plate", "screwdriver"] # default prompts
		prompts = prompt_msg.get("prompts", prompts)

		# Run SAM3 segmentation
		width, height = image.size
		inference_state = processor.set_image(image)
		processor.reset_all_prompts(inference_state)

		merged_mask = None

		try:
			masks_to_merge = []

			for prompt in prompts:
				# Run inference for each prompt
				inference_state_prompt = processor.set_text_prompt(state=inference_state, prompt=prompt)

				# Optional: show masks
				if show_mask:
					img0 = Image.fromarray(img)
					plot_results(img0, inference_state_prompt, show_plot=show_mask)

				# Collect masks for this prompt
				for i in range(len(inference_state_prompt["masks"])):
					mask = inference_state_prompt["masks"][i].squeeze(0).cpu().numpy()
					masks_to_merge.append(mask)

			if len(masks_to_merge) == 0:
				socket.send_json({"error": "no masks found"})
			else:
				# Merge all masks (logical OR)
				merged_mask = np.zeros_like(masks_to_merge[0], dtype=np.uint8)
				for m in masks_to_merge:
					merged_mask = np.logical_or(merged_mask, m)

				merged_mask = merged_mask.astype(np.uint8)  # convert to 0/1
				mask_bytes = merged_mask.tobytes()

				reply_header = json.dumps({
					"height": merged_mask.shape[0],
					"width": merged_mask.shape[1],
					"dtype": str(merged_mask.dtype)
				}).encode("utf-8")

				socket.send_multipart([reply_header, mask_bytes])
				print(f"Sent merged mask back: {merged_mask.shape}")

		except Exception as e:
			print("Error sending merged mask:", e)


if __name__ == "__main__":
    main()