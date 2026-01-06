# To run the live version of sam3

## This live version has the client send the image size in the header with img in bytes, and a variable number of prompts.

1. Follow the instructions for installation in README-BASE.md
2. Once in the conda environment run 
```bash
pip install -r requirements.txt
```
3. To start the server run
```bash
python3 live/server.py
```

### An example server send looks like this
```python
cv_img = self.bridge.imgmsg_to_cv2(self.img, "bgr8")
img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
h, w, c = img_rgb.shape
img_bytes = img_rgb.tobytes()

header = {"height": h, "width": w, "channels": c}

prompts = ["black end effector on robot arm", "cube", "plate", "screwdriver"]
prompts_json = json.dumps({"prompts": prompts}).encode("utf-8")

self.socket.send_multipart([
	json.dumps(header).encode("utf-8"),
	img_bytes,
	prompts_json
])
```

## Note about prompt structure
- prompts is a list of strings or other lists
- if the prompt is a list itll take the first mask, then search for the second and mask it continuously for each prompt in the list for refinement
- if it is a string then simply mask it.

### In `live/` I have added an example ros client. Easily adaptable for strictly python, or any other needs


