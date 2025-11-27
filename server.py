import argparse
import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
image_processor = None
context_len = None
args = None

def load_model(model_path, model_base, load_8bit, load_4bit, device):
    global model, tokenizer, image_processor, context_len
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    
    # Handle generation config renaming logic from predict.py
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'), generation_config)
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, 
            model_base, 
            model_name, 
            load_8bit=load_8bit, 
            load_4bit=load_4bit, 
            device=device
        )
    finally:
        # Restore generation config
        if generation_config is not None and os.path.exists(generation_config):
            os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))

@app.route('/', methods=['GET'])
def index():
    return "Server is running. Send POST requests to /predict"

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    prompt_text = request.form.get('prompt', 'Extract the text in this screenshot and Describe it.')
    
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    # Construct prompt
    qs = prompt_text
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = args.conv_mode
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set pad token id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    device = model.device
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(device) if device.type != 'cpu' else image_tensor.unsqueeze(0).to(device),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    return jsonify({'response': outputs})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-fastvithd_1.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} to {device}...")
    try:
        load_model(args.model_path, args.model_base, args.load_8bit, args.load_4bit, device)
        print("Model loaded.")
        print(f"Server starting on {args.host}:{args.port}")
        print("Registered routes:")
        print(app.url_map)
        app.run(host=args.host, port=args.port)
    except Exception as e:
        print(f"Error loading model or starting server: {e}")
