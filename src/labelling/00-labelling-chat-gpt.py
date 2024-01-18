import json
from openai import OpenAI
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import os 
from icecream import ic 
import random
import re

def openai_fn(msg_system,msg_user):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": msg_system },
        {"role": "user", "content":  msg_user}
    ],
    )
    return completion.choices[0].message.content

# DATA_PATH = "/Users/rahulmehta/Desktop/RESEARCH-2024/Semeval-SHROOM/datasets/SHROOM_dev-v2"
# DATASET = "val.model-agnostic.json"
# DATASET = "val.model-aware.v2.json"

 # op file 
# op_file = "val.model-agnostic-labelled.json"

DATASET_TYPE = 'test'
DATA_PATH = "/Users/rahulmehta/Desktop/RESEARCH-2024/Semeval-SHROOM/datasets/SHROOM_test-unlabeled"
DATASET="test.model-agnostic.json"
op_file = "test.model-agnostic-labelled.json"

# DATASET="test.model-aware.json"
# op_file = "test.model-aware-labelled.json"

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
if __name__=="__main__":
    client = OpenAI()

    # Read df
    file_name = os.path.join(DATA_PATH,DATASET)
    f = open(file_name)
    data = json.load(f)
    
    # Iterating through the json
    # list

    #data = random.sample(data,50)
    ic(len(data))


   
    file_name = os.path.join(DATA_PATH,op_file)
    cnt=0
    with open(file_name,'w',encoding='utf8') as f:
        for item in data:
            cnt+=1
            
            #ic(item["hyp"],item["ref"],item["src"],item["tgt"],item["task"])

            if item["task"] == "MT":
                msg_system = "You are a machine translation system. Reference is either source or target. Is the text in hypothesis a correct English translation of the text in source. Answer in 2 words 1.Yes or no and the 2. probability score "
            elif item["task"] == "DM":
                msg_system = "You are a definition modeling system. Reference is the target. Is the text in hypothesis a correct defintion of the text in source.  Answer in 2 words 1.Yes or no and the 2. probability score "
            elif item["task"] == "PG":
                msg_system = "You are a paraphrase generation system. Reference is either source or target. Is the text in hypothesis a correct paraphrase of the text in source. Answer in 2 words 1.Yes or no and the 2. probability score"
            else:
                pass  
            #msg_user = "hypothesis is I thought you were pregnant ,target is I thought you were pregnant and source is Я думал, ты беременна"
            msg_user = "hypothesis is " +  item["hyp"] + ", target is " + item["tgt"] +  "source is " + item["src"]
            
            response = openai_fn(msg_system,msg_user)
            splits = re.split(', |. ', response)
            ic(cnt,splits)
            label_gpt = 0 if splits[0] == 'Yes' else 1
            

            
            if len(splits)==1:
                probability_gpt = ''
            if len(splits)==2:
                if splits[1]=='':
                     probability_gpt=0
                # elif  splits[1][-1]=='.':   
                #     if is_float(splits[1][0:-1]):
                #         probability_gpt = 1 -float(splits[1][0:-1])
                #     else:
                #         probability_gpt = ""
                elif splits[1][-1]=="%":
                    if is_float(splits[1][0:-1]):
                        probability_gpt = 1 - float(splits[1][0:-1])/100
                    else:
                        probability_gpt = ""
                elif bool(re.search(r'\d', splits[1])):
                    s = re.sub(r'[^\d\.]+', '',splits[1])
                    if s[-1]=='.': 
                        probability_gpt = 1 -float(s[0:-1])
                    else:
                        probability_gpt = 1 -float(s)
                else:
                    probability_gpt = ''
            elif len(splits)>2:
                for s in splits:
                    if is_float(s):
                        probability_gpt = 1 -float(s)
                        break
                    else:
                        probability_gpt = ''
            else:
                probability_gpt = ''

            if DATASET_TYPE == "test":
                ic(label_gpt,probability_gpt)
                item["label"] = label_gpt
                item["p(Hallucination)"] = probability_gpt
            elif DATASET_TYPE == "val":
                ic(response,item['label'],item['p(Hallucination)'],label_gpt,probability_gpt)
                item["label_gpt"] = label_gpt
                item["probability_gpt"] = probability_gpt
            else:
                pass
        
            json.dump(item, f,ensure_ascii=False)
            f.write(os.linesep)

    

