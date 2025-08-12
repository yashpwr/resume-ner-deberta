import os, re, json
from pathlib import Path
from sklearn.model_selection import train_test_split

def kill_surrogates(obj):
    """Remove Unicode surrogate pairs that can cause JSON parsing issues."""
    _SURRO = re.compile(r'[\ud800-\udfff]')
    if isinstance(obj, str): 
        return _SURRO.sub('', obj)
    if isinstance(obj, list): 
        return [kill_surrogates(x) for x in obj]
    if isinstance(obj, dict): 
        return {k: kill_surrogates(v) for k,v in obj.items()}
    return obj

def dump_jsonl(path, rows):
    """Write data to JSONL format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(kill_surrogates(r), ensure_ascii=False) + "\n")

def main():
    # 1) Load Kaggle dataset
    import kagglehub
    path = kagglehub.dataset_download("yashpwrr/resume-ner-training-dataset")
    fp = os.path.join(path, "train.json")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"train.json not found under {path}")

    txt = open(fp, "r", encoding="utf-8", errors="replace").read()
    txt = re.sub(r'\\uD[0-9A-Fa-f]{3}', '', txt)
    data = json.loads(txt)

    # 2) Normalize
    recs = []
    for it in data:
        t = it.get("text","")
        anns = it.get("annotations", [])
        if isinstance(anns, str):
            try: 
                anns = json.loads(anns)
            except: 
                anns = []
        norm = []
        for s,e,lab in anns or []:
            try: 
                s=int(s); e=int(e)
            except: 
                continue
            if 0 <= s <= e <= len(t):
                norm.append([s,e,lab])
        if t:
            recs.append({"text": t, "annotations": norm, "source": "kaggle"})

    # 3) Splits
    train, temp = train_test_split(recs, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    dump_jsonl("data/train.jsonl", train)
    dump_jsonl("data/val.jsonl", val)
    dump_jsonl("data/test.jsonl", test)

    # 4) Ensure artifacts dirs
    os.makedirs("artifacts/logs", exist_ok=True)

    # 5) Train (auto-resume in train.py)
    os.system("python -m src.train --config configs/train.yaml")

if __name__ == "__main__":
    main()
