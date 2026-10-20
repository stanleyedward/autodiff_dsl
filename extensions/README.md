add ext here
1. removing extra atomics -> check in an atomic_add(z[i], x) i is thread_id if yes we can remove atmoic add since its exclusive

2. reduce stack size by removing storage of linear ops, ie if linear theres no need to store intermediates. (TBR)

3.
