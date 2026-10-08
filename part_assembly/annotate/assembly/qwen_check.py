"""Step 10. Check every turn instruction against the renders with a local Qwen vision-language model (we used
Qwen3.8-27B). Per chain, one generation: for each turn the state before the turn and after it (three views each,
from <chain>/img), the changed region outlined in red, and the current instruction. The model says whether the instruction describes exactly what was added
(count, object-centric left/right, identity and colour, attachment) and proposes a corrected sentence when not.
Writes <chain>/instructions_check.json. Flagged turns are then decided by a person (keep / take the proposal / rewrite)
and written to the recipe with apply_instructions.py.

    python qwen_check.py [--source instructions_relational.json] [--claim]
"""
import argparse, json, os, re, sys, time
from paths import hosts as host_list, chain_dir

OUT = 'instructions_check.json'
RULES = """You audit the per-turn instructions of an assembly editing chain of a 3D object for a benchmark. The object is assembled turn by turn from its own parts. For each turn you get: the state BEFORE the turn (three views), the state AFTER the turn (three views), a panel where the region that changed is outlined in red, and the instruction currently written for that turn.
Decide for every turn whether the instruction describes exactly what the images show being added. Check, in this order:
1. Count: how many separate new parts appear? Does the instruction name that many (e.g. it says two arms but only one arm is new because the other was already present before)?
2. Identity: are the new parts what the instruction calls them (arm, wheel, shoe, roof...) with the right colour/material?
3. Side: LEFT/RIGHT must be the object's OWN left and right (decide the object's front from its face, headlights, screen...; the object's left is on the viewer's right when the object faces the viewer). If the object has no intrinsic front, the instruction must not use left/right at all.
4. Placement: does the instruction say what the new part attaches to and where, and is that what the images show?
Trust the images over any text. If the images do not show a detail, say so instead of guessing.
Answer with ONLY a JSON object, no prose, no code fence:
{"front": "<one sentence: which feature is the front, or 'no intrinsic front'>",
 "turns": [{"n": 1, "ok": true|false, "issues": ["<short issue>", ...], "text": "<the instruction as it should read: one imperative sentence, two at most, only the parts actually added in this turn, object-centric left/right or none; copy the current text unchanged when ok>", "zh": "<natural Chinese rendering of text>"}, ...]}
with exactly one entry per turn, in order."""


def strip(d, t, out):
    """Three views (f0000-f0002) of state t side by side."""
    from PIL import Image
    p = f'{out}/strip_turn{t:02d}.png'
    if not os.path.exists(p):
        ims = [Image.open(f'{d}/img/turn{t:02d}/f{v:04d}.png').convert('RGBA') for v in range(3)]
        w, h = ims[0].size; s = Image.new('RGBA', (3 * w, h), (255, 255, 255, 255))
        for i, im in enumerate(ims):
            s.alpha_composite(im, (i * w, 0))
        s.convert('RGB').save(p)
    return p


def region(d, n, out):
    """The state after turn n (view f0000) with the region that changed since turn n-1 outlined in red."""
    import numpy as np
    from PIL import Image, ImageDraw
    p = f'{out}/region_turn{n:02d}.png'
    if not os.path.exists(p):
        a = np.asarray(Image.open(f'{d}/img/turn{n - 1:02d}/f0000.png').convert('RGBA'), np.int16)
        b = Image.open(f'{d}/img/turn{n:02d}/f0000.png').convert('RGBA'); bn = np.asarray(b, np.int16)
        d_rgb = np.abs(a[..., :3] - bn[..., :3]).max(-1); d_a = np.abs(a[..., 3] - bn[..., 3])
        ys, xs = np.nonzero((d_a > 64) | (d_rgb > 96))       # new silhouette or clearly new colour; ignores soft shadow changes
        bg = Image.new('RGBA', b.size, (255, 255, 255, 255)); bg.alpha_composite(b); im = bg.convert('RGB')
        if len(xs):
            m = 6
            ImageDraw.Draw(im).rectangle([max(0, xs.min() - m), max(0, ys.min() - m), min(im.width - 1, xs.max() + m),
                                          min(im.height - 1, ys.max() + m)], outline=(255, 0, 0), width=3)
        im.save(p)
    return p


def build(oid, source):
    d = chain_dir(oid); b = json.load(open(f'{d}/brief.json')); turns = json.load(open(f'{d}/{source}'))['turns']
    tmp = f'{d}/check_inputs'; os.makedirs(tmp, exist_ok=True)
    content = [{'type': 'text', 'text': RULES + f"\n\nObject: {b['host_caption']}\n\nFinal state, four views:"}]
    for p in b['final_state_views']:
        content.append({'type': 'image', 'image': p})
    for t in turns:
        n = t['n']; adds = '; '.join(q['caption'].rstrip('.') for q in b['turns'][n - 1]['adds'])
        content.append({'type': 'text', 'text': f"\n=== Turn {n}\nCurrent instruction: {t['text']}\nPart captions from the asset library for this turn (may be wrong, the images decide): {adds}\nState BEFORE turn {n} (three views):"})
        content.append({'type': 'image', 'image': strip(d, n - 1, tmp)})
        content.append({'type': 'text', 'text': f"State AFTER turn {n} (three views):"})
        content.append({'type': 'image', 'image': strip(d, n, tmp)})
        content.append({'type': 'text', 'text': 'Changed region outlined in red:'})
        content.append({'type': 'image', 'image': region(d, n, tmp)})
    content.append({'type': 'text', 'text': f"\nNow write the JSON with {len(turns)} turn entries."})
    return b['chain'], turns, [{'role': 'user', 'content': content}]


def parse(txt, n):
    txt = re.sub(r'<think>.*?</think>', '', txt, flags=re.S).strip(); m = re.search(r'\{.*\}', txt, flags=re.S); d = json.loads(m.group(0))
    assert len(d['turns']) == n, f'{len(d["turns"])} turns, expected {n}'
    for i, t in enumerate(d['turns']):
        assert 'ok' in t and t.get('text'), f'turn {i + 1} missing ok/text'
    return d


def main():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='Qwen/Qwen3.8-27B'); ap.add_argument('--source', default='instructions_relational.json')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--force', action='store_true'); ap.add_argument('--max-new', type=int, default=3000)
    ap.add_argument('--claim', action='store_true'); a = ap.parse_args()
    hosts = host_list()[:a.limit or None]
    def claim(h):
        if not a.claim:
            return True
        try:
            os.close(os.open(f'{chain_dir(h)}/.check_lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
        except FileExistsError:
            return False
    todo = [h for h in hosts if a.force or not os.path.exists(f'{chain_dir(h)}/{OUT}')]
    print(f'{len(todo)} chains open', flush=True)
    if not todo:
        return
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForMultimodalLM.from_pretrained(a.model, dtype=torch.bfloat16).to('cuda:0').eval()
    for h in todo:
        d = chain_dir(h)
        if (not a.force and os.path.exists(f'{d}/{OUT}')) or not claim(h):
            continue
        t0 = time.time(); out = None; err = ''
        chain, turns, msgs = build(h, a.source)
        for attempt in range(2):
            if attempt:
                msgs.append({'role': 'user', 'content': [{'type': 'text', 'text': f'Your previous answer could not be parsed ({err}). Output only the JSON object with exactly {len(turns)} turn entries.'}]})
            try:
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt', enable_thinking=False)
            except TypeError:
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
            inputs = inputs.to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
            txt = proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
            open(f'{d}/instructions_check_raw.txt', 'w', encoding='utf-8').write(txt)
            try:
                out = parse(txt, len(turns)); break
            except Exception as e:
                err = str(e)[:200]
        if out is None:
            print(f'FAIL {h[:8]}: {err}', flush=True); continue
        res = dict(chain=chain, host_oid=h, model=a.model, checked_at=time.strftime('%Y-%m-%d %H:%M'), front=out.get('front', ''),
                   turns=[dict(n=t['n'], current=t['text'], ok=bool(o.get('ok')), issues=o.get('issues') or [], text=o['text'].strip(), zh=(o.get('zh') or '').strip())
                          for t, o in zip(turns, out['turns'])])
        json.dump(res, open(f'{d}/{OUT}', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'OK {h[:8]} {len(turns)} turns, {sum(1 for t in res["turns"] if not t["ok"])} flagged, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
