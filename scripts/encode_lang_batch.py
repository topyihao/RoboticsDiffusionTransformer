import os
import json

import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 1
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
TARGET_DIR = "data/datasets/put_marker_into_box"
OUTPUT_DIR = "outs/lang_embeddings"  # Directory to save embeddings

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if instruction file exists directly in TARGET_DIR
    instr_file = os.path.join(TARGET_DIR, 'expanded_instruction_gpt-4-turbo.json')
    if os.path.exists(instr_file):
        print(f"Found instruction file directly in target directory: {instr_file}")
        try:
            with open(instr_file, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            
            # Get instructions
            instructions = [instruction_dict['instruction']]
            if 'simplified_instruction' in instruction_dict:
                instructions += instruction_dict['simplified_instruction']
            if 'expanded_instruction' in instruction_dict:
                instructions += instruction_dict['expanded_instruction']
            
            print(f"Found {len(instructions)} instructions to encode")
            
            # Encode the instructions
            tokenized_res = tokenizer(
                instructions, return_tensors="pt",
                padding="longest",
                truncation=True
            )
            tokens = tokenized_res["input_ids"].to(device)
            attn_mask = tokenized_res["attention_mask"].to(device)
            
            with torch.no_grad():
                text_embeds = text_encoder(
                    input_ids=tokens,
                    attention_mask=attn_mask
                )["last_hidden_state"].detach().cpu()
            
            attn_mask = attn_mask.cpu().bool()
            
            # Save the embeddings for training use
            for i, instruction in enumerate(instructions):
                text_embed = text_embeds[i][attn_mask[i]]
                # Save both to task directory and to output directory
                
                # 1. Save to task directory
                save_path = os.path.join(TARGET_DIR, f"lang_embed_{i}.pt")
                torch.save(text_embed, save_path)
                print(f"Saved embedding to {save_path}")
                
                # 2. Save to output directory with instruction as filename
                # Clean the instruction to use as filename (remove punctuation, spaces, etc.)
                clean_instr = instruction.strip()
                if clean_instr:  # Only save if instruction is not empty
                    save_path = os.path.join(OUTPUT_DIR, f"{clean_instr}.pt")
                    torch.save(text_embed, save_path)
                    print(f"Saved embedding to {save_path}")
            
            return  # Exit after processing if file was found directly
        except Exception as e:
            print(f"Error processing instruction file: {e}")
            import traceback
            traceback.print_exc()
    
    # Original code for nested directory structure
    print("Looking for instructions in nested directory structure...")
    task_paths = []
    for sub_dir in os.listdir(TARGET_DIR):
        middle_dir = os.path.join(TARGET_DIR, sub_dir)
        if os.path.isdir(middle_dir):
            for task_dir in os.listdir(middle_dir):
                task_path = os.path.join(middle_dir, task_dir)
                if os.path.isdir(task_path):
                    task_paths.append(task_path)
    
    print(f"Found {len(task_paths)} task paths in nested structure")
    
    # For each task, encode the instructions
    for task_path in tqdm(task_paths):
        try:
            # Load the instructions corresponding to the task from the directory
            instr_file = os.path.join(task_path, 'expanded_instruction_gpt-4-turbo.json')
            if not os.path.exists(instr_file):
                print(f"No instruction file found at {instr_file}, skipping")
                continue
                
            with open(instr_file, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            
            instructions = [instruction_dict['instruction']]
            if 'simplified_instruction' in instruction_dict:
                instructions += instruction_dict['simplified_instruction']
            if 'expanded_instruction' in instruction_dict:
                instructions += instruction_dict['expanded_instruction']
        
            # Encode the instructions
            tokenized_res = tokenizer(
                instructions, return_tensors="pt",
                padding="longest",
                truncation=True
            )
            tokens = tokenized_res["input_ids"].to(device)
            attn_mask = tokenized_res["attention_mask"].to(device)
            
            with torch.no_grad():
                text_embeds = text_encoder(
                    input_ids=tokens,
                    attention_mask=attn_mask
                )["last_hidden_state"].detach().cpu()
            
            attn_mask = attn_mask.cpu().bool()

            # Save the embeddings for training use
            for i, instruction in enumerate(instructions):
                text_embed = text_embeds[i][attn_mask[i]]
                # Save to task directory
                save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
                torch.save(text_embed, save_path)
                
                # Also save to central output directory with instruction as filename
                clean_instr = instruction.strip()
                if clean_instr:  # Only save if instruction is not empty
                    save_path = os.path.join(OUTPUT_DIR, f"{clean_instr}.pt")
                    torch.save(text_embed, save_path)
                    
        except Exception as e:
            print(f"Error processing {task_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
