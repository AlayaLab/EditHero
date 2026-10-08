# Revision of the draft instructions (LLM agent prompt)

Step 7 of the assembly-chain pipeline: one agent per chain checks and revises the Qwen draft (qwen_draft.py) against
the geometry and the renders (we used Claude Haiku 4.5 agents in Claude Code). Fill in `{CHAIN_DIR}`
(<work>/chains/<oid>). Output: {CHAIN_DIR}/instructions_final.json.

---

You are checking and revising the per-turn editing instructions of one assembly chain of a 3D object (the object is assembled turn by turn from its own parts). A local vision model wrote a first draft; your job is to verify every sentence against the geometry and the images and fix what is wrong. Keep what is right.

Files (read both with the Read tool):
- brief: {CHAIN_DIR}/brief.json   (host caption, axes, each turn's added parts with caption / centre_xyz / extent_xyz / touches, the parts already present before the turn, image paths)
- draft: {CHAIN_DIR}/instructions_qwen.json   (the draft's "front" statement and one {n, parts, text, zh} per turn)

Procedure:
1. Look at the four final-state views (brief.final_state_views) with the Read tool. Decide the object's FRONT yourself (face, headlights, door, muzzle, screen...). Compare with the draft's "front"; if it is wrong, correct it. Then fix the object's OWN left/right as an axis mapping, using the centre_xyz of clearly lateral parts: the object's left is on the viewer's right when the object faces the viewer. Write the mapping down in "front" (e.g. "front faces -z; object's left is +x").
2. For each turn, look at its image_after and check the draft sentence against: (a) the listed parts' centre_xyz / extent_xyz / touches, (b) the front and left/right mapping, (c) the parts already present.
   - Every left/right/front/back/top/bottom word must agree with the coordinates under the mapping. A part with centre z far behind the front is at the back, whatever the caption says.
   - Part identity: captions can be wrong (e.g. a "sword hilt" at shoulder height that touches the torso is an upper arm). Name the part by what its position, size, contacts and the image show it to be.
   - The sentence must add exactly the listed parts and nothing else, must not mention turn numbers or history, and must say what the new parts attach to among the parts already present.
   - Keep colour/material words that help identify parts; drop speculation.
   - No left/right at all when the object has no intrinsic front (table, plant, symmetric tower, abstract structure): use relations to existing parts instead.
3. Chinese "zh": natural Chinese as a Chinese speaker would say it (not word-by-word), with 左/右 in the same object-centric meaning.

Write the result to {CHAIN_DIR}/instructions_final.json with this exact shape:
{"chain": ..., "host_oid": ..., "front_qwen": "<draft front>", "front": "<your front + mapping>",
 "turns": [{"n": 1, "parts": [...], "text_qwen": "<draft text>", "zh_qwen": "<draft zh>", "text": "<final text>", "zh": "<final zh>", "changed": true|false, "reason": "<short reason when changed, else empty>"}, ...]}
One entry per turn, in order, "parts" copied from the brief. The draft strings must be copied verbatim into text_qwen / zh_qwen even when you change nothing. Then return a short summary (chain, ok, number of turns, number changed, front).
