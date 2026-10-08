# How the instructions are written

Every record in a chain's `place_report.json` has two sentences: `instruction_original`, the template the engine wrote
from the part captions ("Add <caption> on <slot>.", "Replace <caption> with <caption>.", "Change the material of ...
to ..."), and `instruction` (with `instruction_zh`), the sentence released in `chains.json`. The released sentences were
rewritten from the templates by an LLM agent (Claude) under the rules below and reviewed by us against the renders. For
the assembly chains of family A the same rules are applied through the prompts in `part_assembly/annotate/assembly/prompts/`.

## Rules

1. **Placement in words a person would use.** Say where a part goes by its relation to parts that are already there: on
   top of, under, beside, behind, in front of, between A and B, at the end of, at the base of, against the wall that ...
   Never use axis words (x, z, "positive x", "along the z axis").
2. **Few left/right.** Use left/right only when no relation works (paired limbs, the second item of a mirrored pair), and
   then it is the object's own left/right, after checking the mirror. Objects that stand on the ground or face a direction
   (animals, characters, vehicles, chairs) get their side and facing described by the scene ("in front of the door",
   "facing the headboard").
3. **No leaks from the part library.** Cut slot names ("the arm zneg of the object", "the wheel left"), phrases from the
   part's source scene ("from the 'Driver 2' video game diorama", "worn by the construction worker figure", "positioned
   under the rustic wooden table"), and broken caption joins ("with the component is a cylindrical ...").
4. **Name parts by their current state.** A part that was replaced or retextured earlier in the chain is referred to by
   its new name and material; outdated "retextured in ..." clauses are removed.
5. **Disambiguate twins.** Parts with the same caption are told apart by side (checked against the mirror) or by a
   visible difference ("the taller tree").
6. **Say only what can be seen.** No invisible details, no "or" alternatives ("headboard or backrest": pick the one the
   render shows). A material instruction is one material scheme (it may be complex: two colours, a pattern, worn), never a
   coverage claim ("covering the whole tire and rim"); colour words are plain ("bright glossy red"). When a baked texture
   differs from the instruction, the instruction is corrected to what was baked.
7. **Bilingual.** Every instruction has a Chinese version (`instruction_zh`) written as a Chinese speaker would say it,
   not translated word by word.
