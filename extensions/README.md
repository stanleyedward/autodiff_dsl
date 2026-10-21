add ext here
1. removing extra atomics -> check in an atomic_add(z[i], x) i is thread_id if yes we can remove atmoic add since its exclusive; Automatic Differentiation of Parallel Loops with Formal Methods” from Huckelheim et al

2. reduce stack size by removing storage of linear ops, ie if linear theres no need to store intermediates. (TBR); s. A paper to read is “Linearity Analysis for Automatic Differentiation” from Strout and Hovland.

3. CHEKCPOINTING, (user-based, allow the programmer to decide what to recompute what to store)
