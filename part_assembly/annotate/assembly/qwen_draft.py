"""Step 6. Draft per-turn instructions with a local Qwen vision-language model (we used Qwen3.8-27B).
One generation per chain: the four final-state views + one view per turn, plus the brief's part captions and
coordinates. Writes <chain>/instructions_qwen.json (and the raw model text next to it).

    CUDA_VISIBLE_DEVICES=k python qwen_draft.py --shard k --nshards 4 [--model Qwen/Qwen3.8-27B] [--limit N]
    python qwen_draft.py --claim        # any number of workers on any nodes; each chain is claimed with a lock file
"""
import argparse, json, os, re, sys, time
from paths import hosts as host_list, chain_dir

RULES = """You annotate one assembly editing chain of a 3D object for a benchmark. The object is assembled turn by turn from its own parts. For EVERY turn, write the instruction a person would give a 3D editing tool to perform exactly that turn's addition, with precise spatial language.

First, from the four final-state views, decide where the object's FRONT is (face, headlights, door, muzzle, screen...). Fix the object's OWN left and right from that: the object's left is on the viewer's right when the object faces the viewer. Use the centre_xyz coordinates of clearly lateral parts to map left/right onto the x or z axis, and apply that mapping consistently to every turn.

Rules for each turn's English "text":
- One imperative sentence (two at most). It describes adding exactly that turn's listed parts and nothing else. No turn numbers, no history: describe the new parts and where they go on what is currently present.
- Name parts by what they are, with the captions' colour/material/shape words when needed. Never use part ids.
- Be spatially precise: what the new part attaches to or rests on, and where on it (top/bottom/front/back/left/right, centre/edge/corner, above/below/between/beside a named existing part). Symmetric pairs: "one on each side of ..." or "on the left and right of ...". Give counts.
- LEFT/RIGHT always mean the object's own left and right (a character's left arm, the truck's left front wheel), never the viewer's. If the object has no intrinsic front (table, plant, symmetric tower, abstract structure), do not use left/right; use relations to existing parts instead, or front/back only if some feature defines them.
- Say less rather than guess when the images do not show a detail.
Chinese "zh": a natural Chinese rendering of the same instruction, as a Chinese speaker would say it; 左/右 keep the same object-centric meaning.

Answer with ONLY a JSON object, no prose, no code fence:
{"front": "<one sentence: which feature is the front and which axis direction it faces, or 'no intrinsic front' and what you used instead>",
 "turns": [{"n": 1, "text": "...", "zh": "..."}, ...]}
with exactly one entry per turn, in order."""

def build_messages(b):
    content = [{'type': 'text', 'text': RULES + f"\n\nObject: {b['host_caption']}\nAxes: {b['axes']}\nCameras: {b['camera']}\n\nFinal state, four views:"}]
    for p in b['final_state_views']: content.append({'type': 'image', 'image': p})
    st = b['start_state']
    content.append({'type': 'text', 'text': f"\nStart state ({st['name']}): " + '; '.join(f"{q['caption']} centre={q['centre_xyz']}" for q in st['parts'])})
    content.append({'type': 'image', 'image': st['image']})
    for t in b['turns']:
        adds = '; '.join(f"[{q['pid']}] {q['caption']} centre={q['centre_xyz']} extent={q['extent_xyz']} touches={q['touches']}" for q in t['adds'])
        present = ', '.join(q['caption'].rstrip('.') for q in t['already_present'])
        content.append({'type': 'text', 'text': f"\nTurn {t['n']} ({t['step_name']}) adds: {adds}\nAlready present before this turn: {present}\nImage after turn {t['n']}:"})
        content.append({'type': 'image', 'image': t['image_after']})
    content.append({'type': 'text', 'text': f"\nNow write the JSON with {len(b['turns'])} turn entries."})
    return [{'role': 'user', 'content': content}]

def parse(txt, n):
    txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.S).strip()
    m = re.search(r'\{.*\}', txt, flags=re.S)
    d = json.loads(m.group(0)); turns = d['turns']
    assert len(turns) == n, f'{len(turns)} turns, expected {n}'
    for i, t in enumerate(turns): assert t.get('text') and t.get('zh'), f'turn {i+1} missing text/zh'
    return d

def main():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    ap = argparse.ArgumentParser(); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--model', default='Qwen/Qwen3.8-27B'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-new', type=int, default=3000); ap.add_argument('--claim', action='store_true'); a = ap.parse_args()
    hosts = host_list()
    if not a.claim:
        hosts = hosts[a.shard::a.nshards]
    if a.limit:
        hosts = hosts[:a.limit]
    def claim(h):
        if not a.claim:
            return True
        try:
            os.close(os.open(f'{chain_dir(h)}/.qwen_lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
        except FileExistsError:
            return False
    todo = [h for h in hosts if a.force or not os.path.exists(f'{chain_dir(h)}/instructions_qwen.json')]
    print(f'{len(todo)} chains open', flush=True)
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16).to('cuda:0').eval()
    for h in todo:
        d = chain_dir(h)
        if (not a.force and os.path.exists(f'{d}/instructions_qwen.json')) or not claim(h):
            continue
        b = json.load(open(f'{d}/brief.json')); t0 = time.time(); out = None; err = ''
        for attempt in range(2):
            msgs = build_messages(b)
            if attempt:
                msgs.append({'role': 'user', 'content': [{'type': 'text', 'text': f'Your previous answer could not be parsed ({err}). Output only the JSON object with exactly {len(b["turns"])} turn entries.'}]})
            try:
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
            except TypeError:
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
            inputs = inputs.to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
            txt = proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            open(f'{d}/instructions_qwen_raw.txt', 'w', encoding='utf-8').write(txt)
            try:
                out = parse(txt, len(b['turns'])); break
            except Exception as e:
                err = str(e)[:200]
        if out is None:
            print(f'FAIL {h[:8]}: {err}', flush=True); continue
        res = dict(chain=b['chain'], host_oid=h, model=a.model, front=out.get('front', ''),
                   turns=[dict(n=t['n'], parts=[q['pid'] for q in t['adds']], text=out['turns'][i]['text'].strip(), zh=out['turns'][i]['zh'].strip()) for i, t in enumerate(b['turns'])])
        json.dump(res, open(f'{d}/instructions_qwen.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'OK {h[:8]} {len(res["turns"])} turns {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
