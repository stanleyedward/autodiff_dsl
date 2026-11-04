# loma source of the renderer. K_MAX and CHECKPOINT are substituted by common.py.
def gaussian_alpha(x : In[float], y : In[float],
                   mx : In[float], my : In[float],
                   lsx : In[float], lsy : In[float], lo : In[float]) -> float:
    sx : float = exp(lsx)
    sy : float = exp(lsy)
    dx : float = (x - mx) / sx
    dy : float = (y - my) / sy
    opac : float = 1.0 / (1.0 + exp(0.0 - lo))
    return opac * exp(-0.5 * (dx * dx + dy * dy))

@simd
def render(px : In[Array[float]], py : In[Array[float]],
           mx : In[Array[float]], my : In[Array[float]],
           lsx : In[Array[float]], lsy : In[Array[float]], lo : In[Array[float]],
           cr : In[Array[float]], cg : In[Array[float]], cb : In[Array[float]],
           n : In[int],
           out_r : Out[Array[float]], out_g : Out[Array[float]], out_b : Out[Array[float]]):
    p : int = thread_id()
    T : float = 1.0
    r : float = 0.0
    g : float = 0.0
    b : float = 0.0
    a : float
    j : int = 0
    while (j < n, max_iter := K_MAX CHECKPOINT):
        a = gaussian_alpha(px[p], py[p], mx[j], my[j], lsx[j], lsy[j], lo[j])
        r = r + T * a * cr[j]
        g = g + T * a * cg[j]
        b = b + T * a * cb[j]
        T = T * (1.0 - a)
        j = j + 1
    out_r[p] = r
    out_g[p] = g
    out_b[p] = b

rev_render = rev_diff(render)
