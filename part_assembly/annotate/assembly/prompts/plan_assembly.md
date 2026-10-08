# Assembly plan of one host (LLM agent prompt)

Used for step 3 of the assembly-chain pipeline: one agent per host, any LLM agent that can read files, look at images
and run a shell command (we used Claude agents in Claude Code). Fill in `{OID}` and `{WORK}`
(PXFORM_ASSEMBLY_WORK), give the text below as the agent's task, then run `check_plan.py`.

---

You plan the assembly order of one PartVerse object for an assembly editing chain: the object starts as its main body and is then assembled step by step from its own parts; each step adds one to three parts. A later stage turns every step into an editing instruction ("add the ..."), so steps must be things a person would naturally add as one action.

Host: {OID}
Read with Bash (cat): {WORK}/hosts/{OID}/parts.json
  (object_caption; axes: y up, x left-right, z front-back, object-normalised; parts: pid, caption, centre_xyz, extent_xyz, area_share, touches = pids this part touches)
Look at the four preview views with the Read tool: {WORK}/preview/img/{OID}/f0000.png f0001.png f0002.png f0003.png

Rules:
1. Step 0 (the start state) = the main body: the structural core the other parts attach to (torso / chassis / trunk / main block). One to three pids, together the largest connected core. Name it.
2. Every following step adds one to three parts that TOUCH something already present (use "touches"). Order outward from the core: big structural parts first, then attachments, then small accessories.
3. Put mirror pairs (left/right boots, both wheels, two ears) in ONE step. Never split one obvious object (a hat and its brim) across steps if they are separate pids: put them together.
4. Every pid of the object appears exactly once across all steps. No pid invented, none left out.
5. 5 to 12 steps in total (including step 0). If the object has more than ~30 parts, group more per step (up to 3) rather than exceeding 12 steps.
6. Name each step with a short noun phrase a person would say ("boots", "left leg armor pieces", "front wheels", "roof panel"); use the captions and the images to name by what the part IS, not by its pid.

Write the plan with a heredoc to {WORK}/plans/{OID}.json as JSON of this exact shape (pids as strings, step 0 first):
{"oid": "{OID}", "steps": [{"pids": ["14"], "name": "armored top"}, {"pids": ["2"], "name": "legs"}, {"pids": ["0", "1", "16"], "name": "boots"}]}
Then verify with: python check_plan.py {OID}
Fix and rewrite until the check prints ok. Return a short summary (oid, written, number of steps, number of parts).
