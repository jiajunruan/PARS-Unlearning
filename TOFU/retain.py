import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    "RULE_NPO": {
        "model_path": "Jiajunruan/NPO-Fix",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/RULE_NPO"
    },
    "Minmax": {
        "model_path": "/users/2/jruan/Probe_unlearning/output",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/Minmax"
    },
    "NPO": {
        "model_path": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr5e-05_beta0.5_alpha1_epoch10",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/NPO"
    },
    "GradDiff": {
        "model_path": "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch5",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/GradDiff"
    },
    "Original": {
        "model_path": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/Original"
    },
    "Retrain": {
        "model_path": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        "eval_dir": "/users/2/jruan/pass-k/saves/eval/Retrain"
    }
}

# Generation Settings
N_GENERATIONS = 1 
MAX_NEW_TOKENS = 256
BATCH_SIZE = 16  # <-- Adjust this based on your GPU memory capacity
GPU_ID = 1

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def eval_cosine_similarity_with_model(gen_outputs, ground_truths, model):
    scores = []
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            gen_str, gt_str = str(gen) if gen else "", str(gt) if gt else ""
            if not gen_str or not gt_str:
                scores.append(0.0)
                continue
            gen_emb = model.encode(gen_str, show_progress_bar=False, convert_to_tensor=True)
            gt_emb = model.encode(gt_str, show_progress_bar=False, convert_to_tensor=True)
            cosine_sim = torch.nn.functional.cosine_similarity(gen_emb, gt_emb, dim=0).item()
            scores.append(float(max(0, cosine_sim)))
    return {'cosine_similarity': scores}

def get_entailment_results_with_pipeline(pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True):
    results = []
    if len(gen_outputs) != len(ground_truths):
        ground_truths = [ground_truths[0]] * len(gen_outputs)

    for i in range(0, len(gen_outputs), bs):
        targets_batch = ground_truths[i:i + bs]
        outputs_batch = gen_outputs[i:i + bs]
        rouge_scores_batch = rouge_scores[i:i + bs]

        data_list = []
        for j in range(len(targets_batch)):
            out_txt, tgt_txt = str(outputs_batch[j] or ""), str(targets_batch[j] or "")
            if not out_txt or not tgt_txt:
                results.append({'label': 'none', 'score': 0.0})
                continue
            
            if not tofu or 'forget' in eval_task:
                data_list.append({'text': out_txt, 'text_pair': tgt_txt})
            else:
                data_list.append({'text': tgt_txt, 'text_pair': out_txt})
        
        if data_list:
            batch_results = pipe(data_list)
            filtered = []
            valid_idx = 0
            for j in range(len(targets_batch)):
                if not (str(outputs_batch[j]) and str(targets_batch[j])):
                    continue
                if rouge_scores_batch[valid_idx] < 0.1:
                    filtered.append({'label': 'none', 'score': 0.0})
                else:
                    filtered.append(batch_results[valid_idx])
                valid_idx += 1
            results.extend(filtered)
        else:
            results.extend([{'label': 'none', 'score': 0.0} for _ in outputs_batch if not (str(_) and str(_))])
            
    return {'entailment_labels': [r['label'] for r in results]}

def process_query_responses(query_data, nli_pipeline, st_model, eval_task="forget", tofu=True):
    ground_truth = query_data.get("ground_truth", "")
    responses = query_data.get("responses", [])

    gen_outputs = [res.get("response", "") for res in responses]
    rougeL_recalls = [res.get("rougeL_recall", 0.0) for res in responses]

    cs_scores = eval_cosine_similarity_with_model(gen_outputs, [ground_truth] * len(gen_outputs), st_model)['cosine_similarity']
    es_labels = get_entailment_results_with_pipeline(nli_pipeline, gen_outputs, [ground_truth] * len(gen_outputs), eval_task, rougeL_recalls, bs=30, tofu=tofu)['entailment_labels']
    es_scores = [1.0 if l == 'entailment' else 0.0 for l in es_labels]

    for i, res in enumerate(responses):
        res["CS"] = cs_scores[i]
        res["ES"] = es_scores[i]

    query_data["Best CS"] = max(cs_scores) if cs_scores else None
    query_data["Avg CS"] = np.mean(cs_scores).item() if cs_scores else None
    query_data["Best ES"] = max(es_scores) if es_scores else None
    query_data["Avg ES"] = np.mean(es_scores).item() if es_scores else None
    query_data["responses"] = responses
    return query_data


# =============================================================================
# PHASE 1: GENERATION (BATCHED)
# =============================================================================

def run_generation_phase():
    print(f"\n=======================================================")
    print(f"🚀 PHASE 1: BATCHED TEXT GENERATION")
    print(f"=======================================================\n")
    
    print("⏳ Loading locuslab/TOFU dataset (retain90)...")
    dataset = load_dataset("locuslab/TOFU", "retain90", split="train")
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for method_label, config in MODELS.items():
        model_path = config["model_path"]
        eval_dir = config["eval_dir"]
        os.makedirs(eval_dir, exist_ok=True)
        output_file = os.path.join(eval_dir, f"generations_n{N_GENERATIONS}.json")
        
        if os.path.exists(output_file):
            print(f"⏭️ Skipping generation for {method_label} (File exists: {output_file})")
            continue
            
        print(f"\n⚙️ Loading model {method_label} from {model_path}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Crucial for batch generation: pad on the left so completion happens on the right
            tokenizer.padding_side = "left" 
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
        except Exception as e:
            print(f"⚠️ Error loading {method_label}: {e}")
            continue

        generated_data = []
        print(f"🧠 Generating responses for {method_label} in batches of {BATCH_SIZE}...")
        
        # Iterate over the dataset in batches
        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=f"Generating {method_label}"):
            batch = dataset[i : i + BATCH_SIZE]
            questions = batch["question"]
            ground_truths = batch["answer"]
            
            prompts = [f"Question: {q}\nAnswer:" for q in questions]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=1.0, 
                    top_p=1.0,
                    do_sample=True,
                    num_return_sequences=N_GENERATIONS,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Slice the outputs to grab ONLY the newly generated tokens (avoids prompt stripping bugs)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_length:]
            
            # Decode only the generated part
            decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # Group responses by question (handles N_GENERATIONS >= 1 elegantly)
            idx = 0
            for j in range(len(prompts)):
                responses = []
                for _ in range(N_GENERATIONS):
                    gen_answer = decoded_outputs[idx].strip()
                    
                    rouge_score = scorer.score(ground_truths[j], gen_answer)
                    rougeL_recall = rouge_score['rougeL'].recall
                    
                    responses.append({
                        "response": gen_answer,
                        "rougeL_recall": rougeL_recall
                    })
                    idx += 1
                    
                generated_data.append({
                    "question": questions[j],
                    "ground_truth": ground_truths[j],
                    "responses": responses
                })
            
        # Free memory between models
        del model
        del tokenizer
        torch.cuda.empty_cache()

        # Save generations
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in generated_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Saved generations to: {output_file}")


# =============================================================================
# PHASE 2: EVALUATION & AGGREGATION
# =============================================================================

def run_semantic_eval_phase(gpu_id=0):
    print(f"\n=======================================================")
    print(f"🧠 PHASE 2: SEMANTIC EVALUATION (CS & ES)")
    print(f"=======================================================\n")
    
    print(f"⏳ Loading Eval Models onto GPU {gpu_id}...")
    st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    nli_pipe = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=device)

    results_summary = []

    for method_label, config in MODELS.items():
        eval_dir = config["eval_dir"]
        if not os.path.exists(eval_dir):
            print(f"⚠️ Directory not found, skipping: {eval_dir}")
            continue
            
        gen_filename = f"generations_n{N_GENERATIONS}.json"
        
        for dirpath, _, filenames in os.walk(eval_dir):
            if gen_filename in filenames:
                input_file = os.path.join(dirpath, gen_filename)
                output_file = os.path.join(dirpath, f"generations_n{N_GENERATIONS}_evaluated.json")
                
                overall_avg_cs, overall_avg_es = 0.0, 0.0
                
                if os.path.exists(output_file):
                    print(f"⏭️ Skipping (Already evaluated): {input_file}")
                    with open(output_file, 'r', encoding='utf-8') as infile:
                        all_avg_cs = []
                        all_avg_es = []
                        for line in infile:
                            data = json.loads(line)
                            if data.get("Avg CS") is not None: all_avg_cs.append(data["Avg CS"])
                            if data.get("Avg ES") is not None: all_avg_es.append(data["Avg ES"])
                        overall_avg_cs = np.mean(all_avg_cs) if all_avg_cs else 0.0
                        overall_avg_es = np.mean(all_avg_es) if all_avg_es else 0.0

                else:
                    print(f"\n⚙️ Evaluating: {input_file}")
                    processed_results = []
                    with open(input_file, 'r', encoding='utf-8') as infile:
                        for line in tqdm(infile.readlines(), leave=False, desc="Evaluating"):
                            try:
                                data = json.loads(line)
                                processed = process_query_responses(data, nli_pipe, st_model, eval_task="retain", tofu=True)
                                processed_results.append(processed)
                            except Exception as e:
                                print(f"Error processing line: {e}")
                                continue

                    if processed_results:
                        all_avg_cs = [res.get("Avg CS", 0) for res in processed_results if res.get("Avg CS") is not None]
                        all_avg_es = [res.get("Avg ES", 0) for res in processed_results if res.get("Avg ES") is not None]
                        
                        overall_avg_cs = np.mean(all_avg_cs) if all_avg_cs else 0.0
                        overall_avg_es = np.mean(all_avg_es) if all_avg_es else 0.0

                        with open(output_file, 'w', encoding='utf-8') as outfile:
                            for data in processed_results:
                                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        
                        print(f"✅ Saved results to: {output_file}")
                
                print(f"📊 {method_label} -> Overall Avg CS: {overall_avg_cs:.4f} | Overall Avg ES: {overall_avg_es:.4f}")
                
                results_summary.append({
                    "Model": method_label,
                    "Avg_CS_Retain": overall_avg_cs,
                    "Avg_ES_Retain": overall_avg_es
                })

    print(f"\n=======================================================")
    print(f"📈 COMPILING RESULTS INTO retain.csv")
    print(f"=======================================================\n")
    
    if results_summary:
        df = pd.DataFrame(results_summary)
        df.to_csv("retain.csv", index=False)
        print("✅ Successfully generated retain.csv!")
        print(df)
    else:
        print("⚠️ No data processed. retain.csv not generated.")

if __name__ == "__main__":
    run_generation_phase()
    run_semantic_eval_phase(gpu_id=GPU_ID)