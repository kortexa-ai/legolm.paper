"""Registered, bounded sensor-plus-instruction generation experiment.

Labels, short sentences, and explicit binary rules have deterministic scoring.
These tasks test controlled generation, not unrestricted multimodal reasoning.
"""
from __future__ import annotations

import random
import re

import torch

from .evaluate import configure, intervene


TEMPLATES = {
    "label": {
        "train": ["Name the sensed {noun}. Reply with only its category.",
                  "What {noun} is sensed? Give the category alone.",
                  "Identify the {noun}; output only the category name."],
        "test": ["Return just the category of the observed {noun}.",
                 "Classify this {noun}. Your entire answer must be its name.",
                 "Which {noun} does the sensor indicate? Category only."],
    },
    "sentence": {
        "train": ["Describe the sensed {noun} using exactly: The {noun} is CATEGORY.",
                  "Write one sentence about the sensor: The {noun} is CATEGORY.",
                  "Report the category in this form: The {noun} is CATEGORY."],
        "test": ["Complete this report with the observed category: The {noun} is CATEGORY.",
                 "Express the sensor reading as: The {noun} is CATEGORY.",
                 "Use the sentence template 'The {noun} is CATEGORY.' for this reading."],
    },
    "rule": {
        "train": ["If the sensed {noun} is {query}, reply {yes}. Otherwise reply {no}. Output one word.",
                  "Apply this rule: {query} means {yes}; any other {noun} means {no}. Reply with that word only.",
                  "Choose {yes} when the {noun} is {query}, and {no} in every other case. One word only."],
        "test": ["For this sensor reading, map category {query} to {yes} and all remaining categories to {no}. Return the mapped word.",
                 "Use the following code: {yes} denotes {query}; {no} denotes any other {noun}. Return just the code.",
                 "Respond with {yes} if category {query} matches the sensor, or {no} if it does not. No explanation."],
    },
}
NOUN = {"imu": "activity", "audio": "sound", "vision": "object"}


def category_text(value):
    return value.replace("_", " ").lower()


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower().replace("_"," ")).rstrip(".!")


def case(dataset, label, family, split, template, *, query, reverse=False):
    category=category_text(dataset["categories"][label])
    noun=NOUN[dataset["modality"]]
    yes,no=("no","yes") if reverse else ("yes","no")
    prompt=TEMPLATES[family][split][template].format(noun=noun,query=category_text(dataset["categories"][query]),yes=yes,no=no)
    target=category if family=="label" else (f"The {noun} is {category}." if family=="sentence" else (yes if label==query else no))
    return {"family":family,"template_split":split,"template":template,"prompt":prompt,"target":target,
            "label":label,"query_label":query,"reverse_rule":reverse}


def chat_ids(tokens, prompt, response=None):
    if tokens.host != "qwen08" or tokens.hf is None:
        raise ValueError("The registered instruction experiment uses Qwen 0.8B's native chat template")
    messages=[{"role":"user","content":prompt}]
    if response is not None:
        messages.append({"role":"assistant","content":response})
    encoded=tokens.hf.apply_chat_template(messages,tokenize=True,return_dict=True,add_generation_prompt=response is None,enable_thinking=False)
    ids=encoded["input_ids"]
    if not isinstance(ids,list) or not all(isinstance(i,int) for i in ids):
        raise TypeError("Expected one flat native-chat token sequence")
    return ids


def training_rows(tokens, dataset, labels, step):
    rows=[]
    masks=[]
    rng=random.Random(17171+step)
    for i,label in enumerate(labels):
        family=("label","sentence","rule")[(step*len(labels)+i)%3]
        query=label if rng.randrange(2)==0 else rng.choice([c for c in range(len(dataset["categories"])) if c!=label])
        item=case(dataset,label,family,"train",rng.randrange(3),query=query,reverse=bool(rng.randrange(2)))
        prompt_ids=chat_ids(tokens,item["prompt"])
        full=chat_ids(tokens,item["prompt"],item["target"])
        if full[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Native chat template does not preserve the assistant response boundary")
        if len(full)>256:
            raise ValueError("Registered generation example exceeds 256 tokens; no implicit truncation")
        rows.append(full)
        masks.append(len(prompt_ids))
    width=max(map(len,rows))-1
    x=torch.zeros((len(rows),width),dtype=torch.long)
    y=torch.full_like(x,-1)
    for i,(row,boundary) in enumerate(zip(rows,masks)):
        x[i,:len(row)-1]=torch.tensor(row[:-1])
        # Each assistant token is predicted exactly once, including native EOS.
        y[i,boundary-1:len(row)-1]=torch.tensor(row[boundary:])
    return x,y


def panel_indices(dataset):
    test=dataset["test"]
    if dataset["modality"]=="audio":
        return list(range(len(test["ids"])))
    # Round-robin over class x subject cells, then class; input IDs are fixed
    # without consulting any host output. Vision is a bounded N5a diagnostic.
    rng=random.Random(700042)
    cells={}
    for i,(label,group) in enumerate(zip(test["labels"],test["groups"])):
        cells.setdefault((label,group),[]).append(i)
    for members in cells.values():
        rng.shuffle(members)
    selected=[]
    keys=sorted(cells)
    while len(selected)<min(300,len(test["ids"])):
        for key in keys:
            if cells[key] and len(selected)<300:
                selected.append(cells[key].pop())
    return selected


def evaluation_cases(dataset, indices):
    rng=random.Random(700043)
    result=[]
    for ordinal,index in enumerate(indices):
        label=dataset["test"]["labels"][index]
        query=label if ordinal%2==0 else rng.choice([c for c in range(len(dataset["categories"])) if c!=label])
        for family in ("label","sentence","rule"):
            for split,template in [("train",0),("test",0),("test",1),("test",2)]:
                for reverse in ([False,True] if family=="rule" else [False]):
                    result.append({"sensor_index":index,**case(dataset,label,family,split,template,query=query,reverse=reverse)})
    return result


def score(response,item,dataset):
    output=normalize(response)
    exact=output==normalize(item["target"])
    if item["family"]=="rule":
        return {"semantic_correct":exact,"format_compliant":output in ("yes","no"),"joint_success":exact}
    names=[category_text(x) for x in dataset["categories"]]
    # Longest phrase wins so "walking upstairs" is not also plain "walking".
    matches=[i for i,name in enumerate(names) if re.search(r"(?<!\w)"+re.escape(name)+r"(?!\w)",output)]
    if matches:
        max_len=max(len(names[i]) for i in matches)
        matches=[i for i in matches if len(names[i])==max_len]
    semantic=matches==[item["label"]]
    noun=NOUN[dataset["modality"]]
    allowed=names if item["family"]=="label" else [normalize(f"The {noun} is {n}.") for n in names]
    return {"semantic_correct":semantic,"format_compliant":output in allowed,"joint_success":exact}


@torch.no_grad()
def decode(host, adaptation, kind, features, prompts, *, native_chat, intervention="correct", max_new_tokens=64):
    """Each call starts a new cache after setting its own sensor-conditioned weights."""
    if kind=="prefix":
        raise ValueError("Prefix generation is outside the registered generation grid")
    configure(host,adaptation,kind,features.to(host.device),intervention=intervention)
    rows=[chat_ids(host.tokens,p) if native_chat else host.tokens.task_encode(p) for p in prompts]
    if host.name=="mini":
        if len(rows)!=1:
            raise ValueError("Mini raw generation uses scalar decoding")
        row=rows[0][:]
        output=[]
        for _ in range(max_new_tokens):
            token=int(host.logits(torch.tensor([row],device=host.device))[:,-1].argmax(-1))
            output.append(token);row.append(token)
            if token==host.tokens.eos:
                break
        return [host.tokens.decode(output)]
    pad=host.tokens.hf.pad_token_id
    if pad is None:
        pad=host.tokens.eos if isinstance(host.tokens.eos,int) else host.tokens.eos[0]
    width=max(map(len,rows))
    x=torch.full((len(rows),width),pad,dtype=torch.long,device=host.device)
    mask=torch.zeros_like(x)
    for i,row in enumerate(rows):
        x[i,-len(row):]=torch.tensor(row,device=host.device);mask[i,-len(row):]=1
    out=host.model.generate(input_ids=x,attention_mask=mask,do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,
                            pad_token_id=pad,use_cache=True)
    return host.tokens.hf.batch_decode(out[:,width:],skip_special_tokens=True)


@torch.no_grad()
def evaluate_generation(host,adaptation,kind,dataset,constant,*,objective,intervention="correct",batch_size=8,oracle_labels=None):
    indices=panel_indices(dataset)
    selected,mapping=intervene(dataset["test"]["features"],dataset["test"]["labels"],intervention,700044,constant.cpu())
    if objective=="G":
        cases=evaluation_cases(dataset,indices)
    else:
        cases=[{"sensor_index":i,"family":"label","template_split":"original-ranking","template":0,
                "prompt":dataset["prompt"],"target":category_text(dataset["categories"][dataset["test"]["labels"][i]]),
                "label":dataset["test"]["labels"][i]} for i in indices]
    items=[]
    for start in range(0,len(cases),batch_size):
        group=cases[start:start+batch_size]
        prompts=[c["prompt"] for c in group]
        if oracle_labels is not None:
            prompts=["The sensor category is "+category_text(dataset["categories"][oracle_labels[c["sensor_index"]]])+". "+c["prompt"] for c in group]
        responses=decode(host,adaptation,kind,selected[[c["sensor_index"] for c in group]],prompts,
                         native_chat=objective=="G",intervention=intervention)
        for c,p,response in zip(group,prompts,responses):
            i=c["sensor_index"]
            items.append({**c,"id":dataset["test"]["ids"][i],"group":dataset["test"]["groups"][i],"actual_prompt":p,
                          "sensor_id":None if mapping is None else dataset["test"]["ids"][mapping[i]],
                          "response":response,**score(response,c,dataset)})
    summaries={}
    for family in sorted({i["family"] for i in items}):
        for split in sorted({i["template_split"] for i in items}):
            rows=[i for i in items if i["family"]==family and i["template_split"]==split]
            if rows:
                summaries[family+"/"+split]={"outputs":len(rows),"sensor_units":len({r["id"] for r in rows}),
                    **{key:sum(r[key] for r in rows)/len(rows) for key in ("semantic_correct","format_compliant","joint_success")}}
    return {"summary":summaries,"items":items,"panel_ids":[dataset["test"]["ids"][i] for i in indices],
            "decoding":{"greedy":True,"max_new_tokens":64,"native_chat":objective=="G","enable_thinking":False},"intervention":intervention}
