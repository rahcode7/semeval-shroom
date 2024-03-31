!CMAKE_ARGS="-DLLAMA_CUBLAS=on " pip install 'llama-cpp-python>=0.1.79' --force-reinstall --upgrade --no-cache-dir
!pip install huggingface_hub
!pip install datasets

from huggingface_hub import hf_hub_download

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

from llama_cpp import Llama
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=16,
    n_batch=8000,
    n_gpu_layers=32,
    n_ctx=8192, 
    logits_all=True
)

run_on_test = False 

path_val_model_aware = "path/to/reference/val.model-aware.json"
path_val_model_aware_output = "path/to/output/val.model-aware.json"

from datasets import load_dataset
import json
import random
import numpy as np
import tqdm.notebook as tqdm

seed_val = 442
random.seed(seed_val)
np.random.seed(seed_val)

with open('/content/train.model-aware.v2.json', 'r') as istr:
    data_val_all = json.load(istr)
num_sample = len(data_val_all)

output_json = []
labels = ["Not Hallucination", "Hallucination"]

for i in tqdm.trange(num_sample):
    
    task=str(data_val_all[i]['task'])
    
     if run_on_test:
        id=int(data_val_all[i]['id'])
        
     hyp=str(data_val_all[i]['hyp'])
     src=str(data_val_all[i]['src'])
     tgt=str(data_val_all[i]['tgt'])

    if task == "PG":
        context = f"Context: {src}"
    else:
        context = f"Context: {tgt}"

    sentence=f"Sentence: {hyp}"
    
     message=f"{context}\n{sentence}\nIs the Sentence supported by the Context above? Answer using ONLY yes or no:"
     
     prompt=f"<s>[INST] {message} [/INST]"

    response=lcpp_llm(
        prompt=prompt,
        temperature= 0.0,
        logprobs=1,
    )
    
     answer=str(response["choices"][0]["text"]).strip().lower()
     
      if answer.startswith("yes"):
         output_label="Not Hallucination"
         prob=1-float(np.exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
         
      elif answer.startswith("no"):
          output_label="Hallucination"
          prob=float(np.exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
          
       else:
           idx_random=random.randint(0,len(labels)-1)
           output_label=labels[idx_random]
           prob=float(0.5)

       item_to_json={"label":output_label, "p(Hallucination)":prob}
       
       if run_on_test:
            item_to_json['id']=id
            
       output_json.append(item_to_json)


f=open(path_val_model_aware_output,'w',encoding='utf-8')
json.dump(output_json,f)
f.close()

path_val_model_agnostic="path/to/reference/val.model-agnostic.json"
path_val_model_agnostic_output="path/to/output/val.model-agnostic.json"

dataset = load_dataset('json', data_files={'val': path_val_model_agnostic})
data_val_all = dataset['val']
num_sample=len(data_val_all)

output_json = []

for i in tqdm.trange(num_sample):
    
    task=str(data_val_all[i]['task'])
    
     if run_on_test:
        id=int0]["logprobs"]["token_logprobs"][0]))
         
      elif answer.startswith("no"):
          output_label="Hallucination"
          prob=float(np.exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
          
       else:
           idx_random=random.randint(0,len(labels)-1)
           output_label=labels[idx_random]
           prob=float(0.5)

       item_to_json={"label":output_label, "p(Hallucination)":prob}
       
       if run_on_test:
            item_to_json['id']=id
            
       output_json.append(item_to_json)


f=open(path_val_model_agnostic_output,'w',encoding='utf-8')
json.dump(output_json,f)
f.close()