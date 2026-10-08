# Relational rewrite of the instructions (LLM agent prompt)

Step 9 of the assembly-chain pipeline, after lr_fix.py: one agent per chain rewrites the placement wording so that it
relies on relations to parts already present instead of axis words and bare left/right (we used Claude Haiku 4.5 agents
in Claude Code). Fill in `{CHAIN_DIR}` and `{OID}`. Output: {CHAIN_DIR}/instructions_relational.json.
The same style rules were applied to the instructions of every family (docs/INSTRUCTIONS.md).

---

You rewrite per-turn editing instructions for a 3D assembly chain (a PartVerse object is assembled part by part; each turn adds one or a few parts to what is already there).

Chain dir: {CHAIN_DIR}
Read with Bash (cat): brief.json (host caption; per-turn parts with caption, centre_xyz, extent_xyz, touches = pids the part touches; start_state; which parts exist before each turn) and instructions_checked.json (current English 'text' and Chinese 'zh' per turn, and 'front' = which axis the object faces and which axis is the object's own left).

Rewrite every turn's text (English) and zh (Chinese). Rules, in priority order:
1. KEEP EVERY DETAIL of the current text: every part, colour, material, count, pattern, and per-item differences (if the old text says "a boot with a brown band on one foot and a boot with pink trim on the other", the new text must still say which boot has which). Do not shorten the content; only change HOW the placement is described. Never invent parts.
2. NEVER write axis words: no x/y/z, no "positive x", "negative z", "+x", "along the z axis".
3. Describe placement by relation to parts that are already present: on top of, under, beside, behind, in front of, between A and B, at the end of, touching, at the base of, in the corner where A meets B, on the same side as C, mirrored on the opposite side of D. Use touches/centres/extents in brief.json to choose relations that are actually true. Prefer such relations over "left/right".
4. "left"/"right" only when nothing else disambiguates: paired body parts of a character (left arm, right foot), or a second item of a mirrored pair ("on the side opposite the existing pouch" is better than "on the right"). When you do write left/right it means the object's own left/right as given by 'front'; do not flip it.
5. One to three sentences per turn, imperative, natural spoken language. Chinese must read like a native speaker wrote it, not a translation (e.g. 靠着前墙并排放, 紧挨着, 在…和…之间, 顶端, 底部, 另一侧); no abbreviations.
6. Write the file {CHAIN_DIR}/instructions_relational.json with a heredoc (JSON, UTF-8, straight double quotes, no trailing commas):
{"chain": "<chain id from instructions_checked.json>", "host_oid": "{OID}", "source": "instructions_checked.json", "rule": "relational placement, no axis words, minimal left/right, all details kept", "turns": [{"n": <int>, "parts": [...same as checked...], "text_checked": "<old text>", "zh_checked": "<old zh>", "text": "<new>", "zh": "<new>"}, ...]}
Include every turn from instructions_checked.json with the same n. Then verify: python3 -c "import json;d=json.load(open('{CHAIN_DIR}/instructions_relational.json'));print(len(d['turns']))" must equal the checked turn count, and grep -ciE "\b(x|y|z)[- ]?axis|(positive|negative) ?[xyz]\b" on the file must be 0. Do not modify any other file. Return a short summary (chain, written, number of turns).
