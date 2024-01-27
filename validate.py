from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

llama_large = AutoModelForCausalLM.from_pretrained('/vast/work/public/ml-datasets/llama-2/Llama-2-13b-hf', cache_dir='/scratch/jp6263/slackV2/hf/models/', torch_dtype=torch.float16).to('cuda')
llama_large_tokenizer = AutoTokenizer.from_pretrained('/vast/work/public/ml-datasets/llama-2/Llama-2-13b-hf/')

def calculate_logprob(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        return loss.item()

# Load responses from JSON
with open('/home/jp6263/other/responses.json', 'r') as file:
    responses = json.load(file)

# Calculate log probabilities using LLaMA
for response in responses:
    text = response['response']
    logprob = calculate_logprob(llama_large, llama_large_tokenizer, text)
    response['llama_logprob'] = logprob
    response['diff'] = logprob - (response['logprob']**2)**0.5
responses = sorted(responses, key=lambda x: x['diff'], reverse=True)
# Print the results
for response in responses:
    print(f"Response: {response['response']}, Llama Logprob: {response['llama_logprob']}")

# Write results to a new JSON file
output_file = '/home/jp6263/other/cotra_llama_responses.json'
with open(output_file, 'w') as file:
    json.dump(responses, file, indent=4)

print(f"LLaMA log probabilities have been saved to {output_file}")