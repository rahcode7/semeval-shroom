#
# This script performs exploratory modeling for Shroom Task 6
# It uses SelfCheckGPT and Vectara models for sequence classification
# The model is trained on a combined dataset of model-aware and model-agnostic data
# The script also includes data preprocessing and splitting into training and validation sets
# Finally, the trained model is saved and used for inference on test data
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)
model = CrossEncoder('vectara/hallucination_evaluation_model')

def ensemble_sgp_vectara(hyp, tgt):
    sgp_score = selfcheck_nli.predict(sentences = [str(hyp)], sampled_passages = [str(tgt)])
    vectara_score = 1 - float(model.predict([str(tgt), str(hyp)]))
    probability = (sgp_score + vectara_score) / 2
    if probability < 0.5:
        label = "Not Hallucination"
    else:
        label = "Hallucination"
    return label, probability[0]

ensemble_sgp_vectara("Having been in the same place for a long time .", "consecutively ; in a row")

def generate_pred(data_val_all, num_sample):
    output_json = []
    labels = ["Not Hallucination", "Hallucination"]
    for i in tqdm.trange(num_sample):
        id_ = i
        task = str(data_val_all[i]['task'])
        hyp = str(data_val_all[i]['hyp'])
        src = str(data_val_all[i]['src'])
        tgt = str(data_val_all[i]['tgt'])
        if task == "PG":
            tgt = src
        output_label, prob = ensemble_sgp_vectara(hyp, tgt)

        item_to_json={"label":output_label, "p(Hallucination)":prob, "id": id_}
        output_json.append(item_to_json)
        
    return output_json



with open('/content/shroom6/train.model-aware.v2.json', 'r') as aware:
    data_val_aware = json.load(aware)

output_json_aware = generate_pred(data_val_aware, len(data_val_aware))



f=open("/kaggle/working/val.model-aware.json",'w',encoding='utf-8')
json.dump(output_json_aware,f)
f.close()



with open('/content/shroom6/train.model-agnostic.json', 'r') as agnostic:
    data_val_agnostic = json.load(agnostic)

output_json_agnostic = generate_pred(data_val_agnostic, len(data_val_agnostic))



f=open("/kaggle/working/val.model-agnostic.json",'w',encoding='utf-8')
json.dump(output_json_agnostic,f)
f.close()