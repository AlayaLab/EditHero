"""Qwen-Image-Edit reference worker using the chain spool protocol.

Jobs: job_*.json {img, prompt, out, seed?}; receipts: .started and .done.
Set PXFORM_RESTYLE_MODEL to use a local snapshot. Run in the diffusion environment.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--spool', required=True)
    parser.add_argument('--model', default=os.environ.get('PXFORM_RESTYLE_MODEL', 'Qwen/Qwen-Image-Edit-2509'))
    parser.add_argument('--idle-timeout', type=float, default=3600)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to('cuda')
    print('[restyle_serve] ready', flush=True)
    last = time.monotonic()
    while not (spool / 'STOP').exists():
        jobs = [p for p in sorted(spool.glob('job_*.json')) if not Path(str(p)+'.done').exists()]
        if not jobs:
            if time.monotonic() - last > args.idle_timeout:
                break
            time.sleep(.3)
            continue
        for path in jobs:
            started = time.monotonic()
            Path(str(path)+'.started').write_text(str(time.time()))
            try:
                job = json.loads(path.read_text())
                image = Image.open(job['img']).convert('RGB')
                image.thumbnail((768, 768))
                output = pipe(image=[image], prompt=job['prompt'], negative_prompt=' ',
                              true_cfg_scale=4., num_inference_steps=int(job.get('steps', 40)),
                              generator=torch.Generator(device='cuda').manual_seed(int(job.get('seed', 0)))).images[0]
                output.save(job['out'])
                message = f'ok {time.monotonic()-started:.1f}'
            except Exception:
                message = 'error ' + traceback.format_exc()
            Path(str(path)+'.done').write_text(message)
            print(path.name, message, flush=True)
            last = time.monotonic()


if __name__ == '__main__':
    main()
