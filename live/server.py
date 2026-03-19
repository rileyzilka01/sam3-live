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
		prompts = ["black end effector on robot arm"] # default prompts
		prompts = prompt_msg.get("prompts", prompts)

		# Run SAM3 segmentation
		width, height = image.size
		inference_state = processor.set_image(image)
		processor.reset_all_prompts(inference_state)

		merged_mask = None

		try:
			send_json = {}
			mask_bytes = []

			for prompt in prompts:
				masks_to_merge = []
				# Run inference for each prompt
				if isinstance(prompt, str):
					masks = run_prompt(inference_state, prompt)

				# Case 2: nested prompt list (sequential masking)
				elif isinstance(prompt, list):
					masks = None

					for idx, sub_prompt in enumerate(prompt):
						sub_mask = run_prompt(inference_state, sub_prompt)

						if sub_mask is None:
							masks = None
							break

						if masks is None:
							# First prompt initializes mask
							masks = sub_mask
						else:
							# Subsequent prompts are masked by previous result
							new_masks = []
							for m1 in masks:
								for m2 in sub_mask:
									new_masks.append(np.logical_and(m1, m2).astype(np.uint8))
							masks = new_masks

				else:
					raise ValueError(f"Unsupported prompt type: {type(prompt)}")

				if masks is None or len(masks) == 0:
					send_json[str(prompt)] = []
					continue

				if prompt == prompts[0]:
					for i in range(len(masks)-1):
						masks[0] = np.logical_or(masks[i], masks[i+1]).astype(np.uint8)
					masks = [masks[0]]

				send_json[str(prompt)] = []
				for m in masks:
					m = m.astype(np.uint8)
					mask_bytes.append(m.tobytes())

					send_json[str(prompt)].append({
						"height": m.shape[0],
						"width": m.shape[1],
						"dtype": str(m.dtype)
					})

			reply_header = json.dumps(send_json).encode("utf-8")
			socket.send_multipart([reply_header] + mask_bytes)
			print(f"Sent masks back for prompts: {send_json.keys()}, {len(mask_bytes)}")

		except Exception as e:
			print("Error sending merged mask:", e)


def run_prompt(state, text_prompt):
	state_prompt = processor.set_text_prompt(
		state=state,
		prompt=text_prompt
	)

	prompt_masks = []
	for i in range(len(state_prompt["masks"])):
		mask = state_prompt["masks"][i].squeeze(0).cpu().numpy()
		prompt_masks.append(mask.astype(np.uint8))  # ensure consistent dtype

	if len(prompt_masks) == 0:
		return None

	if text_prompt == "robot arm" or text_prompt == "black object":
		merged = np.zeros_like(prompt_masks[0], dtype=bool)
		for m in prompt_masks:
			merged = np.logical_or(merged, m)
		prompt_masks = [merged]

	return prompt_masks 

if __name__ == "__main__":
    main()