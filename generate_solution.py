import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import xmltodict
import spacy
import scispacy
import en_core_sci_sm
from scispacy.abbreviation import AbbreviationDetector

import random
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--submission_file_path', type=str, default='./data/test/cb_submission.json') # ./data/test/mb_submission.json
parser.add_argument('--dataset_file_path', type=str, default='./data/test/archehr-qa.xml')
# Thresholds
parser.add_argument('--srmed_eps', type=float, default=0.0) # 0.9
parser.add_argument('--cbmed_eps', type=float, default=0.4) # 0.05
parser.add_argument('--mces_eps', type=float, default=0.9) # 0.7
parser.add_argument('--cptce_eps', type=float, default=0.5) # 0.5
# Models
parser.add_argument('--mces_model_path', type=str, default='../../models/Bio_ClinicalBERT') # ../../models/bert-base-multilingual-cased
parser.add_argument('--medcptce_model_path', type=str, default='../../models/MedCPT-Cross-Encoder')
parser.add_argument('--med42_model_path', type=str, default='../../models/Llama3-Med42-8B')


parser.add_argument('--seed', type=int, help='random seed', default=42)




##########################################
### Max Cosine Entity Similarity Score ###
##########################################

def get_max_cosine_score(question_ents, sentences_ents, mces_model, mces_tokenizer):
    scores = []
    for i, sentence_ents in enumerate(sentences_ents):
        q_scores = []
        for q_ent in question_ents:
            sen_scores = []
            for s_ent in sentence_ents:
                emb_q = get_embedding(q_ent, mces_model, mces_tokenizer)
                emb_s = get_embedding(s_ent, mces_model, mces_tokenizer)
                cos_sim = F.cosine_similarity(emb_q, emb_s, dim=0).item()
                sen_scores.append(cos_sim)
            q_scores.append(max(sen_scores) if sen_scores else 0.0)
        scores.append(np.mean(sorted(q_scores, reverse=True)[:1]) if q_scores else 0.0)
    return scores

    

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, hidden_size)
        attention_mask = inputs['attention_mask']

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        summed = masked_embeddings.sum(1)
        counts = mask.sum(1)
        mean_pooled = summed / counts
        return mean_pooled.squeeze()



##################################
### MedCPT Cross Encoder Score ###
##################################

def get_medcpt_embedding(pairs, medcptce_model, medcptce_tokenizer): 
    with torch.no_grad():
        encoded = medcptce_tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        logits = medcptce_model(**encoded).logits.squeeze(dim=1)
        probs = (logits - logits.min()) / (logits.max() - logits.min())
    return probs

def rank_sentences_by_similarity(question, sentences, medcptce_model, medcptce_tokenizer):
    pairs = [[question, sentence] for sentence in sentences]
    return get_medcpt_embedding(pairs, medcptce_model, medcptce_tokenizer)



###########################################
### Context Based Med42 Few Shot Scores ###
###########################################

def cb_compute_response_score(question, sentence, context, med42_model, med42_tokenizer):
    few_shot = """You are a medical assistant helping a patient's family member understand the discharge summary. The family member asks a general question about the patient’s condition or expected recovery. From the discharge summary, you are evaluating whether a specific sentence is essential to help them understand what they truly need to know — even if they didn't ask about it directly.

For each example, decide:
- Is the sentence important for answering the underlying concern in the question? ("Yes" or "No")
- Briefly explain why or why not.

### Example 1
Context:
The patient was admitted with signs of dehydration and electrolyte imbalance following several days of vomiting and diarrhea. Intravenous fluids and potassium replacement were administered. He gradually regained strength and tolerated oral intake by day 3. There were no signs of infection. Electrolyte levels normalized. He was encouraged to maintain oral hydration and avoid NSAIDs. Discharge instructions included dietary recommendations. He is to follow up with his primary care physician in one week. The patient lives alone and has limited mobility. Transportation services were arranged for follow-up.

Patient’s Question: How long will it take for him to fully recover?
Sentence: "He is to follow up with his primary care physician in one week."
Answer: Yes
Reason: The scheduled follow-up provides insight into the expected timeline of recovery and monitoring, even though the patient didn't explicitly ask about appointments.

### Example 2
Context:
The patient presented with acute asthma exacerbation. She received nebulized albuterol and corticosteroids in the emergency department. Oxygen saturation improved over 24 hours. There were no signs of pneumonia. She was discharged with a prescription for inhaled corticosteroids and a tapering dose of prednisone. She was advised to avoid known triggers such as smoke or allergens. Patient reported improved breathing at rest but slight shortness of breath during activity. No further imaging was ordered. The pulmonologist will review her progress in 10 days.

Patient’s Question: Is she okay to go back to work next week?
Sentence: "The pulmonologist will review her progress in 10 days."
Answer: Yes
Reason: The timing of the specialist review is crucial for determining readiness to return to work, even though the patient didn't mention the appointment.

### Example 3
Context:
The patient was admitted for routine laparoscopic cholecystectomy. The surgery was uncomplicated. Minimal intraoperative bleeding was noted. Postoperative pain was managed with oral analgesics. Bowel function resumed within 24 hours. She ambulated independently on post-op day 2. The surgical wound was clean and dry. Discharge instructions advised avoiding heavy lifting for two weeks. Follow-up scheduled with surgery clinic in 14 days. Patient was in good spirits and eager to return to normal activities.

Patient’s Question: What should her recovery look like?
Sentence: "Discharge instructions advised avoiding heavy lifting for two weeks."
Answer: Yes
Reason: The lifting restriction is an essential part of understanding the expected recovery process, even if not directly requested.

### Example 4
"""
    # Build full prompt
    #prompt = few_shot.strip() + "\n\n" + f"Context: {sentence}\nQuestion: What information is essential from this context for answering the question \"{question}\"\nAnswer:"
    prompt = few_shot.strip() + "\n" + f"Context:\n{context}\n\nPatient’s Question: {question}\nSentence: \"{sentence}\"\nAnswer:"
    input_ids = med42_tokenizer(prompt, return_tensors="pt").input_ids.to(med42_model.device)
    with torch.no_grad():
        outputs = med42_model(input_ids)
        logits = outputs.logits[:, -1, :]  # only the next token

    # Tokenize input
    inputs = med42_tokenizer(prompt, return_tensors="pt").to(med42_model.device)
    
    # Generate model output
    with torch.no_grad():
        outputs = med42_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=med42_tokenizer.pad_token_id
        )
    
    # Decode generated answer
    generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
    generated_answer = med42_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    
    # Compute log-probability score
    answer_ids = med42_tokenizer(" " + generated_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(med42_model.device)
    logits = torch.stack(outputs.scores, dim=1)[0]  # shape: [tokens, vocab]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    
    # Sum log-probs for each token
    score = sum(log_probs[i, token_id].item() for i, token_id in enumerate(answer_ids[0]))
    
    if generated_answer[:3] == "Yes":
        return math.exp(score)
    else:
        return 0.0


def cbmed42_predict(question, sentences, med42_model, med42_tokenizer):
    scores = []
    for i, sen in enumerate(sentences):
        scores.append(cb_compute_response_score(question, " ".join(sen.split()), " ".join("\n".join(sentences).split()), med42_model, med42_tokenizer))
    return list(np.array(scores)/sum(scores))




###############################################
### Sentence Relevant Med42 Few Shot Scores ###
###############################################

def sr_compute_response_score(question, sentence, med42_model, med42_tokenizer):
    few_shot = """You are a clinical assistant. Given a context and a question, extract only the essential information from the context that is necessary to answer the question. If no information is relevant, respond with "None". Also provide a short explanation for your answer.

Context: The patient has a history of hypertension and presents with progressive shortness of breath. BNP levels are elevated. Physical examination reveals bilateral rales and mild pedal edema.
Question: What information is essential from this context for answering the question "What is causing the patient's breathing difficulty?"
Answer: elevated BNP, bilateral rales, mild pedal edema
Reason: These are all indicators of congestive heart failure, which likely explains the breathing difficulty.

Context: The patient is a 45-year-old male with a history of allergic rhinitis. He was seen in allergy clinic and placed on a regimen of nasal corticosteroids and antihistamines. No new triggers identified. Symptoms are seasonal and well-controlled.
Question: What information is essential from this context for answering the question "What is the most likely cause of the patient's anemia?"
Answer: None
Reason: The context is entirely focused on allergic rhinitis, with no hematologic data or symptoms of anemia.

Context: The patient completed a dental cleaning and X-rays showed mild periodontal disease. Oral hygiene habits were discussed, and the patient agreed to floss daily. No pain or bleeding reported. No antibiotics were prescribed.
Question: What information is essential from this context for answering the question "What medications are responsible for the patient's elevated INR?"
Answer: None
Reason: There is no mention of any anticoagulants or medications that affect coagulation in the context.

Context: Patient underwent knee replacement two years ago. Reports occasional clicking sensation but no pain. X-ray shows proper implant positioning.
Question: What information is essential from this context for answering the question "Is the knee replacement causing complications?"
Answer: occasional clicking sensation, no pain, proper implant positioning
Reason: Clicking may suggest minor mechanical noise but no signs of complications given the lack of pain and good positioning.

Context: Complains of weight loss and fatigue over the past 3 months. Labs show iron deficiency anemia. Colonoscopy reveals a 2 cm mass in the ascending colon.
Question: What information is essential from this context for answering the question "What might explain the patient’s fatigue?"
Answer: iron deficiency anemia, 2 cm mass in ascending colon
Reason: Chronic blood loss from the mass could explain anemia and fatigue.

Context: The patient underwent cataract surgery on the right eye and reports improved vision. Post-op evaluation showed clear lens placement and normal intraocular pressure. No inflammation noted. Scheduled for left eye surgery in two months.
Question: What information is essential from this context for answering the question "Why did the patient develop shortness of breath?"
Answer: None
Reason: The context is limited to ophthalmologic findings and does not mention any pulmonary or cardiovascular symptoms.

Context: Denies smoking, alcohol, or drug use. Family history positive for lung cancer in both parents. Works in construction for 25 years without respiratory protection.
Question: What information is essential from this context for answering the question "What are the patient's risk factors for lung cancer?"
Answer: family history of lung cancer, 25 years in construction without respiratory protection
Reason: Occupational exposure and genetics increase risk even without smoking.

Context: Admitted for severe epigastric pain. Has history of NSAID use for chronic back pain. Labs show decreased hemoglobin. Endoscopy confirms a gastric ulcer.
Question: What information is essential from this context for answering the question "What is the likely cause of the gastrointestinal bleeding?"
Answer: NSAID use, gastric ulcer, decreased hemoglobin
Reason: NSAIDs are known to cause gastric ulcers, which can lead to bleeding.

Context: No prior psychiatric history. The patient has been irritable and withdrawn for the past month. Sleep has decreased to 3 hours/night. Appetite remains normal.
Question: What information is essential from this context for answering the question "Are there signs of depression?"
Answer: irritability, social withdrawal, decreased sleep
Reason: These are common symptoms associated with depressive disorders.

Context: The patient had a colonoscopy last week, which revealed three polyps that were removed. Pathology is pending. The patient denies abdominal pain, nausea, or changes in bowel habits. Family history is negative for colorectal cancer.
Question: What information is essential from this context for answering the question "Why is the patient experiencing chronic fatigue?"
Answer: None
Reason: The context is focused on GI screening and doesn't include symptoms, labs, or findings that would explain fatigue.

Context: Presents with left arm weakness and facial droop for 45 minutes. Symptoms resolved prior to arrival. CT scan shows no acute infarct. History of atrial fibrillation.
Question: What information is essential from this context for answering the question "What might have caused the neurological symptoms?"
Answer: transient symptoms, atrial fibrillation
Reason: AFib can cause transient ischemic attacks, which present with stroke-like symptoms that resolve.

Context: Mother reports that her child, aged 3, has not yet started speaking in full sentences. Hearing test is normal. No social interaction issues observed. Growth chart is appropriate.
Question: What information is essential from this context for answering the question "Is there concern for developmental delay?"
Answer: 3-year-old not speaking in full sentences
Reason: While social and hearing are normal, speech delay is suggestive of possible developmental delay.

Context: Recent travel to sub-Saharan Africa. Developed intermittent fever and chills on return. Blood smear reveals Plasmodium falciparum.
Question: What information is essential from this context for answering the question "What is the likely cause of the patient’s fever?"
Answer: travel to sub-Saharan Africa, Plasmodium falciparum
Reason: These findings point to malaria as the likely cause of the fever.

Context: Complains of morning stiffness lasting more than 1 hour. Joints in both hands are swollen and tender. Positive rheumatoid factor and anti-CCP antibodies.
Question: What information is essential from this context for answering the question "Is this likely to be rheumatoid arthritis?"
Answer: morning stiffness >1 hour, swollen/tender hand joints, positive RF and anti-CCP
Reason: These clinical and serological findings are diagnostic of RA.

Context: A 65-year-old woman was referred to audiology due to recent hearing difficulties. Audiogram showed moderate bilateral sensorineural hearing loss. Hearing aids were recommended. No signs of vertigo or tinnitus were reported.
Question: What information is essential from this context for answering the question "What led to the patient's episodes of syncope?"
Answer: None
Reason: The context only contains auditory assessment and does not address cardiovascular or neurologic causes.

Context: On insulin therapy. Skipped lunch due to meetings. Found diaphoretic and confused. Glucose 42 mg/dL.
Question: What information is essential from this context for answering the question "What explains the patient’s confusion?"
Answer: skipped lunch, insulin therapy, glucose 42 mg/dL
Reason: Hypoglycemia is likely due to missed meal with insulin use.

Context: Reports worsening shortness of breath over 2 weeks. Has COPD. Oxygen saturation drops to 89% on ambulation. Chest X-ray shows no infiltrates.
Question: What information is essential from this context for answering the question "What is likely contributing to the patient’s shortness of breath?"
Answer: COPD history, desaturation with ambulation
Reason: COPD with exertional desaturation is a common cause of dyspnea in such patients.

Context: Diagnosed with hypothyroidism last year. Currently on levothyroxine. Complains of fatigue and cold intolerance. TSH 9.2.
Question: What information is essential from this context for answering the question "Why is the patient still symptomatic?"
Answer: hypothyroidism, TSH 9.2
Reason: Elevated TSH indicates under-replacement with levothyroxine.

Context: Denies any chest pain. Takes beta-blocker for hypertension. EKG reveals bradycardia (HR 48 bpm). Patient feels fatigued.
Question: What information is essential from this context for answering the question "What could explain the fatigue?"
Answer: beta-blocker use, bradycardia
Reason: Bradycardia from beta-blockers may result in reduced cardiac output and fatigue.

Context: The patient was evaluated in the ophthalmology clinic due to complaints of blurry vision. Examination showed no signs of diabetic retinopathy. Blood pressure was within normal range. There were no neurological deficits noted. Follow-up was scheduled in six months.
Question: What information is essential from this context for answering the question "What is the underlying cause of the patient's persistent headaches?"
Answer: None
Reason: The context only discusses ophthalmological findings and vision-related complaints but contains no information about the cause of headaches.

Context: Patient presented for a follow-up regarding their post-operative shoulder surgery. Physical therapy was recommended and patient reports improvement in range of motion. There are no signs of infection or complications. Sleep has improved as well.
Question: What information is essential from this context for answering the question "What factors contributed to the patient's recent weight loss?"
Answer: None
Reason: The context only discusses orthopedic recovery and makes no mention of diet, metabolism, or weight.

Context: The patient was brought in for confusion. No focal neurological deficits noted. BUN and creatinine significantly elevated. Recently started lisinopril.
Question: What information is essential from this context for answering the question "What could explain the altered mental status?"
Answer: elevated BUN/creatinine, started lisinopril
Reason: Acute kidney injury from ACE inhibitors may lead to uremic encephalopathy.

Context: 65-year-old with chronic low back pain. MRI shows mild degenerative disc disease. No nerve compression.
Question: What information is essential from this context for answering the question "Is surgery indicated?"
Answer: mild degenerative disc disease, no nerve compression
Reason: Conservative treatment is favored as no surgical lesion is present.

Context: During the dermatology consultation, the patient described new-onset skin lesions. The rash appeared on the arms and back, non-pruritic and non-painful. No signs of infection were noted. Biopsy was scheduled.
Question: What information is essential from this context for answering the question "Why has the patient developed elevated liver enzymes?"
Answer: None
Reason: The context centers around dermatological symptoms with no hepatic or metabolic findings provided.

Context: History of mechanical heart valve replacement. INR today is 5.2. No active bleeding reported.
Question: What information is essential from this context for answering the question "What explains the elevated INR?"
Answer: mechanical valve replacement
Reason: Patients require anticoagulation for valves, which can overshoot and elevate INR.

Context: Breast mass noted on exam. Mammogram shows suspicious lesion. Biopsy confirms ductal carcinoma in situ.
Question: What information is essential from this context for answering the question "What is the diagnosis?"
Answer: ductal carcinoma in situ
Reason: Biopsy provides definitive diagnosis.

Context: Patient with ESRD on dialysis. Missed last two sessions. Complains of generalized weakness. Potassium level is 6.8.
Question: What information is essential from this context for answering the question "What is the likely cause of weakness?"
Answer: missed dialysis sessions, potassium 6.8
Reason: Hyperkalemia and uremia due to missed dialysis likely explain weakness.
"""
    # Build full prompt
    prompt = few_shot.strip() + "\n\n" + f"Context: {sentence}\nQuestion: What information is essential from this context for answering the question \"{question}\"\nAnswer:"
    input_ids = med42_tokenizer(prompt, return_tensors="pt").input_ids.to(med42_model.device)
    with torch.no_grad():
        outputs = med42_model(input_ids)
        logits = outputs.logits[:, -1, :]  # only the next token

    # Tokenize input
    inputs = med42_tokenizer(prompt, return_tensors="pt").to(med42_model.device)
    
    # Generate model output
    with torch.no_grad():
        outputs = med42_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=med42_tokenizer.pad_token_id
        )
    
    # Decode generated answer
    generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
    generated_answer = med42_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Compute log-probability score
    answer_ids = med42_tokenizer(" " + generated_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(med42_model.device)
    num_generated = len(outputs.scores)
    answer_ids = answer_ids[:, :num_generated]  # truncate if longer


    logits = torch.stack(outputs.scores, dim=1)[0]  # shape: [tokens, vocab]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    
    # Sum log-probs for each token
    score = sum(log_probs[i, token_id].item() for i, token_id in enumerate(answer_ids[0]))
    
    if generated_answer[:4] != "None":
        return math.exp(score)
    else:
        return 0.0

def srmed42_predict(question, sentences, med42_model, med42_tokenizer):
    scores = []
    for i, sen in enumerate(sentences):
        scores.append(sr_compute_response_score(question, " ".join(sen.split()), med42_model, med42_tokenizer))
    return list(np.array(scores)/sum(scores))


####################################
### Submission Answer Generation ###
####################################

def concisely_paraphrase(sentence, med42_model, med42_tokenizer):
    few_shot = """You are a clinical assistant specialized in simplifying discharge summaries.
Your task is to take a long clinical sentence and rewrite it as a shorter, natural, and concise sentence that preserves the essential clinical information.
Do not copy the entire sentence or use unnecessary detail. Keep it factual, clear, and brief.

Sentence: The patient was admitted to the hospital due to a sudden episode of chest pain that occurred while he was gardening.
Compressed: Admitted for sudden chest pain during gardening.

Sentence: Following the MRI scan, the patient was found to have a small herniated disc at the L4-L5 level.
Compressed: MRI showed a small herniated disc at L4-L5.

Sentence: The patient has a medical history of hypertension, type 2 diabetes, and chronic kidney disease stage 3.
Compressed: History includes hypertension, diabetes, and stage 3 kidney disease.

Sentence: She was prescribed albuterol inhaler to be used as needed for episodes of shortness of breath.
Compressed: Prescribed albuterol for shortness of breath as needed.

Sentence: During his hospital stay, the patient developed a mild skin rash likely due to a reaction to antibiotics.
Compressed: Developed mild rash from antibiotics.

Sentence: The patient was advised to follow a low-sodium diet and monitor blood pressure regularly at home.
Compressed: Advised low-sodium diet and home blood pressure monitoring.

Sentence: He lives alone but receives weekly assistance from his daughter with groceries and medication management.
Compressed: Lives alone with weekly help from daughter.

Sentence: The patient’s vaccination record was updated during the follow-up visit, including influenza and tetanus boosters.
Compressed: Received flu and tetanus boosters at follow-up.
"""
    prompt = few_shot + f"\n\nSentence: {sentence}\nCompressed:"

    # Tokenization and generation
    inputs = med42_tokenizer(prompt, return_tensors="pt").to(med42_model.device)
    max_tokens = 50

    with torch.no_grad():
        outputs = med42_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=med42_tokenizer.pad_token_id
        )

    # Decode
    generated_text = med42_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].split("\n")[0].strip()
    return answer

def compress_sentence(question, sentence, med42_model, med42_tokenizer, max_answer_words=15):
    few_shot = """You are a clinical assistant helping family members understand discharge summaries.
Your task is to answer questions based on long clinical sentences, which may include irrelevant information.
Always provide a direct, natural answer that is as concise as possible.
Do not repeat or copy any part of the question in your answer.
Do not begin the answer with phrases like “Because...” or “XYZ was recommended because...”.
If no clear answer is possible, reply with: None

Question: What treatment did the patient receive for pneumonia?
Sentence: The patient was diagnosed with pneumonia and treated with intravenous antibiotics and oxygen therapy.
Answer: He was treated with antibiotics and oxygen therapy.

Question: Why is the patient taking insulin?
Sentence: Due to a recent diagnosis of type 2 diabetes, the patient was prescribed insulin to manage blood sugar levels.
Answer: He was diagnosed with type 2 diabetes.

Question: What caused the patient's shortness of breath?
Sentence: The patient's shortness of breath was likely due to fluid accumulation in the lungs caused by heart failure.
Answer: He had lung fluid from heart failure.

Question: What mobility assistance does the patient need?
Sentence: After hip surgery, the patient requires a walker and supervision while moving.
Answer: He requires a walker and supervision.

Question: Why was a walking cane recommended to the patient?
Sentence: The patient’s vaccination record was updated during the follow-up visit, including influenza and tetanus boosters.
Answer: None

Question: What complications occurred during the patient's hospital stay?
Sentence: The patient experienced atrial fibrillation, transient confusion, and a mild allergic reaction to antibiotics during admission.
Answer: He experienced atrial fibrillation, confusion, and an allergic reaction.
"""
    prompt = few_shot + f"\n\nQuestion: {question}\nSentence: {sentence}\nAnswer:"

    # Tokenization and generation
    inputs = med42_tokenizer(prompt, return_tensors="pt").to(med42_model.device)
    max_tokens = 50

    with torch.no_grad():
        outputs = med42_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=med42_tokenizer.pad_token_id
        )

    # Decode
    generated_text = med42_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].split("\n")[0].strip()
    if answer[:4] == "None":
        answer = concisely_paraphrase(sentence, med42_model, med42_tokenizer)
    return answer





def main(args):


    ####################
    ### Load dataset ###
    ####################


    nlp = en_core_sci_sm.load()
    nlp.add_pipe("abbreviation_detector")

    def extract_clinical_terms(text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents]


    logging.info("loading dataset")

    with open(args.dataset_file_path, 'r', encoding='utf-8') as f:
        dev_data = xmltodict.parse(f.read())

    processed_dev_data = []
    for case_id in range(len(dev_data["annotations"]["case"])):
        question_en = extract_clinical_terms(dev_data["annotations"]["case"][case_id]["clinician_question"]) #patient_narrative  #clinician_question
        question = dev_data["annotations"]["case"][case_id]["clinician_question"]
        context = dev_data["annotations"]["case"][case_id]["note_excerpt"]
        sentences = []
        sentences_en = []
        sentence_offsets = [case_id]
        sentences_ids = []
        for sentence in dev_data["annotations"]["case"][case_id]["note_excerpt_sentences"]["sentence"]:
            sentences_ids.append({
                "id": sentence["@id"],
                "text": sentence["#text"]
            })
            sentence_offsets.append(context.find(sentence["#text"], sentence_offsets[-1]))
            sentences_en.append(extract_clinical_terms(sentence["#text"]))
            sentences.append(sentence["#text"])
        processed_dev_data.append({
            "question": question,
            "question_ents": question_en,
            "question_full": question,
            "sentences_ents": sentences_en,
            "sentences_full": sentences,
            "sentences_ids": sentences_ids,
            "case_id": dev_data["annotations"]["case"][case_id]["@id"]
        })



    # Max Cosine Entity Similarity Score
    mces_tokenizer = AutoTokenizer.from_pretrained(args.mces_model_path)
    mces_model = AutoModel.from_pretrained(args.mces_model_path)

    # Collect scores
    mces_scores = []
    for i, ex in enumerate(processed_dev_data):
        mces_scores.append(get_max_cosine_score(ex["question_ents"], ex["sentences_ents"], mces_model, mces_tokenizer))
        logging.info(f"\t{i}. sample processed")

    logging.info("Max Cosine Entity Similarity Score computed")

    # MedCPT Cross Encoder Score

    medcptce_tokenizer = AutoTokenizer.from_pretrained(args.medcptce_model_path)
    medcptce_model = AutoModelForSequenceClassification.from_pretrained(args.medcptce_model_path)

    # Collect scores
    medcptce_scores = []
    for i, ex in enumerate(processed_dev_data):
        medcptce_scores.append(rank_sentences_by_similarity(", ".join(ex["question_ents"]), [", ".join(sen_en) for sen_en in ex["sentences_ents"]], medcptce_model, medcptce_tokenizer))
        logging.info(f"\t{i}. sample processed")

    logging.info("MedCPT Cross Encoder Score computed")

    # Context Based Med42 Few Shot Scores

    # Load model
    med42_tokenizer = AutoTokenizer.from_pretrained(args.med42_model_path)
    med42_model = AutoModelForCausalLM.from_pretrained(args.med42_model_path, device_map="auto", torch_dtype=torch.float16)
    med42_model.eval()


    # Collect scores
    cbmed42_scores = []
    for i, ex in enumerate(processed_dev_data):
        cbmed42_scores.append(cbmed42_predict(ex["question_full"], ex["sentences_full"], med42_model, med42_tokenizer))
        logging.info(f"\t{i}. sample processed")

    logging.info("Context Based Med42 Few Shot scores computed")

    # Sentence Relevant Med42 Few Shot Scores

    # Collect scores
    srmed42_scores = []
    for i, ex in enumerate(processed_dev_data):
        srmed42_scores.append(srmed42_predict(ex["question_full"], ex["sentences_full"], med42_model, med42_tokenizer))
        logging.info(f"\t{i}. sample processed")


    logging.info("Sentence Relevant Med42 Few Shot Scores processed")



    # GET FINAL PREDICTION BY COMBINING ALL SCORES
    logging.info("Combining all scores ...")
    final_pred = []
    for srmed_sample_scs, cbmed_sample_scs, mces_sample_scs, cptce_sample_scs in zip(srmed42_scores, cbmed42_scores, mces_scores, medcptce_scores):
        srmed_preds = [1 if sc > args.srmed_eps else 0 for sc in srmed_sample_scs]
        cbmed_preds = [1 if sc > args.cbmed_eps else 0 for sc in cbmed_sample_scs]
        mces_preds = [1 if sc > args.mces_eps else 0 for sc in mces_sample_scs]
        cptce_preds = [1 if sc > args.cptce_eps else 0 for sc in cptce_sample_scs]
        preds = [a | b | c | d for a, b, c, d in zip(srmed_preds, cbmed_preds, mces_preds, cptce_preds)]
        assert sum(preds) > 0
        final_pred.append(preds)
    logging.info(final_pred)
    logging.info("Evidence Sentences are predicted successfully")



    # GENERATE SUBMISSION ANSWERS

    max_answer_words = 75
    final_submission = []
    for ex_id, (ex, pred) in enumerate(zip(processed_dev_data, final_pred)):
        sentence_ids = []
        compressed_sentences = []
        full_sentences = []
        one_sentence_limit = int(max_answer_words/sum(pred))
        
        # Collect compressed sentences
        for i, label in enumerate(pred):
            if label == 1:
                compressed_sentences.append(compress_sentence(ex["question_full"], " ".join(ex["sentences_ids"][i]["text"].split()), med42_model, med42_tokenizer, one_sentence_limit-1))
                sentence_ids.append(ex["sentences_ids"][i]["id"])
                full_sentences.append(" ".join(ex["sentences_ids"][i]["text"].split()))

        # Ensure the max answer words limit
        current_longest = 0
        prev_length = -1
        current_length = sum([len(sen.split()) for sen in compressed_sentences])
        while sum([len(sen.split()) for sen in compressed_sentences]) > max_answer_words:
            long_sen_id = np.argsort([len(sen.split()) for sen in compressed_sentences])[::-1][current_longest]
            new_sen = concisely_paraphrase(compressed_sentences[long_sen_id], med42_model, med42_tokenizer)
            if len(new_sen.split()) < len(compressed_sentences[long_sen_id].split()):
                compressed_sentences[long_sen_id] = new_sen
            if current_length == sum([len(sen.split()) for sen in compressed_sentences]):
                current_longest += 1
            else:
                current_length = sum([len(sen.split()) for sen in compressed_sentences])
            if current_longest == len(compressed_sentences):
                current_longest = 0
                if prev_length == current_length:
                    # Remove last word from the longest sentence
                    longest_for_removing_word_id = np.argsort([len(sen.split()) for sen in compressed_sentences])[::-1][0]
                    compressed_sentences[longest_for_removing_word_id] = " ".join(compressed_sentences[longest_for_removing_word_id].split()[:-1]) + "."
                    logging.info(len(compressed_sentences[longest_for_removing_word_id].split(" ")))
                    logging.info("Warning: Last word of the longest sentence was removed")
                prev_length = current_length
                current_length = sum([len(sen.split()) for sen in compressed_sentences])
                logging.info(current_length)
                logging.info("Warning: Unable to shorten sentences. Retrying...")
        
        
        # If the length is small, use original sentences - start with the smallest ones
        sorted_indices = sorted(range(len(full_sentences)), key=lambda i: len(full_sentences[i]))
        for short_sentence_i in sorted_indices:
            if sum([len(sen.split()) for sen in compressed_sentences]) - len(compressed_sentences[short_sentence_i].split()) + len(full_sentences[short_sentence_i].split()) <= max_answer_words:
                compressed_sentences[short_sentence_i] = full_sentences[short_sentence_i]
            else:
                break

        # Collect compressed sentences to one answer string with cites
        answer = [com_sen + " |" + sen_id + "|" for com_sen, sen_id in zip(compressed_sentences, sentence_ids)]
        sample_submission = {
            "case_id" : ex["case_id"],
            "answer": "\n".join(answer)
        }
        
        logging.info(f"\t{ex_id}. submission answer generated")
        logging.info(len(sample_submission["answer"].split(' ')))
        final_submission.append(sample_submission)

    logging.info(final_submission)


    # Save submission answer to a file

    with open(args.submission_file_path, "w", encoding="utf-8") as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=2)



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
