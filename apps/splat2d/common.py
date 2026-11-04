# Shared helpers: compile the loma renderer, call it on the GPU, Adam.
import io, contextlib, os, sys
here = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(here))
sys.path.append(root)
import numpy as np
import slangpy
from PIL import Image
import compiler
import slang_utils
import _asdl.loma as loma_ir
from extensions import pipeline

PARAMS = ['mx', 'my', 'lsx', 'lsy', 'lo', 'cr', 'cg', 'cb']
OFF = dict(exclusive=False, tbr=False, checkpoint=False, privatize=False, patch_slang=True)

# name: (pipeline flags, checkpointed source)
CONFIGS = {
    'baseline':               (dict(OFF), False),
    '+privatize':             (dict(OFF, privatize=True), False),
    '+privatize +exclusive':  (dict(OFF, privatize=True, exclusive=True), False),
    '+privatize +TBR':        (dict(OFF, privatize=True, tbr=True), False),
    '+privatize +checkpoint': (dict(OFF, privatize=True, checkpoint=True), True),
    'all':                    (dict(OFF, privatize=True, exclusive=True, tbr=True, checkpoint=True), True),
}


def source(k_max, ck=None):
    with open(os.path.join(here, 'splat.py')) as f:
        code = f.read()
    return (code.replace('K_MAX', str(k_max))
                .replace(' CHECKPOINT', f', checkpoint := {ck}' if ck else ''))


def stack_slots(funcs):
    """Per-thread stack slots of the reverse pass: rev_render plus every _d_rev_ helper it calls."""
    return sum(s.t.static_size
               for name, f in funcs.items() if name == 'rev_render' or name.startswith('_d_rev_')
               for s in f.body if isinstance(s, loma_ir.Declare) and s.target.startswith('_t_'))


class Renderer:
    """render / rev_render compiled to Slang for one config, for up to k_max Gaussians."""
    def __init__(self, config, k_max, ck=16):
        flags, checkpointed = CONFIGS[config]
        pipeline.enable(**flags)
        # one device per compile: slangpy caches modules by the name '_tmp'
        self.device = slang_utils.create_slang_device()
        with contextlib.redirect_stdout(io.StringIO()):
            _, self.kernels = compiler.compile(source(k_max, ck if checkpointed else None),
                                               target='slang', slang_device=self.device)
        self.funcs = pipeline.last_funcs
        self.stack_slots = stack_slots(self.funcs)
        pipeline.disable()

    def call(self, name, values, threads):
        """Calls `name` with In arguments from `values` (keyed by loma argument name).
        Out arrays and adjoints are allocated here; returns them keyed by name,
        adjoints as 'd' + primal name (e.g. 'dmx')."""
        args, outs = self._args(name, values, threads)
        self._bind(name, args, threads)()
        self.wait()
        ids = {id(a): i for i, a in enumerate(args)}
        return {k: self._buffers[ids[id(v)]].to_numpy() for k, v in outs.items()}

    def bind(self, name, values, threads):
        """Uploads the arguments once and returns a function that dispatches `name`
        on them (no wait), for timing the kernel without the uploads."""
        return self._bind(name, self._args(name, values, threads)[0], threads)

    def _args(self, name, values, threads):
        args, outs = [], {}
        for a in self.funcs[name].args:
            if a.i == loma_ir.In():
                args.append(values[a.id])
                continue
            # Out argument: a primal output (out_r) or an adjoint (_dmx_XXXXXX)
            key = a.id if not a.id.startswith('_d') else 'd' + a.id[2:].rsplit('_', 1)[0]
            if isinstance(a.t, loma_ir.Array):
                size = threads if not a.id.startswith('_d') else len(values[key[1:]])
                outs[key] = np.zeros(size, dtype=np.float32)
            else:
                outs[key] = np.zeros(1, dtype=np.int32)
            args.append(outs[key])
        return args, outs

    def _bind(self, name, args, threads):
        kernel = self.kernels[name]
        p = list(kernel.program.layout.find_function_by_name(name).parameters)
        self._buffers = [slangpy.Tensor.from_numpy(device=self.device, ndarray=a)
                         if isinstance(a, np.ndarray) else a for a in args]
        kw = {'thread_count': [threads, 1, 1], p[1].name: threads}
        for param, v in zip(p[2:], self._buffers):
            kw[param.name] = v.storage if isinstance(v, slangpy.Tensor) else v
        return lambda: kernel.dispatch(**kw)

    def wait(self):
        for name in ('wait_for_idle', 'wait'):
            if hasattr(self.device, name):
                return getattr(self.device, name)()

    def render(self, inputs, params):
        h, w = inputs['shape']
        out = self.call('render', {**inputs, **params, 'n': len(params['mx'])}, h * w)
        return np.stack([out['out_r'], out['out_g'], out['out_b']], -1).reshape(h, w, 3)

    def rev_values(self, inputs, params, dout):
        """Arguments of rev_render; dout is the (H, W, 3) adjoint of the rendered image."""
        dout = dout.reshape(-1, 3).astype(np.float32)
        return {**inputs, **params, 'n': len(params['mx']),
                'out_r': np.ascontiguousarray(dout[:, 0]),
                'out_g': np.ascontiguousarray(dout[:, 1]),
                'out_b': np.ascontiguousarray(dout[:, 2])}

    def grads(self, inputs, params, dout):
        h, w = inputs['shape']
        out = self.call('rev_render', self.rev_values(inputs, params, dout), h * w)
        return {k: out['d' + k] for k in PARAMS}


def pixel_coords(h, w):
    py, px = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return {'px': px.ravel(), 'py': py.ravel(), 'shape': (h, w)}


def init_params(k, h, w, seed=0):
    """Means uniform over the image; std-dev sqrt(h*w/k), so k Gaussians roughly tile it."""
    rng = np.random.default_rng(seed)
    f = lambda v: np.full(k, v, dtype=np.float32)
    log_std = 0.5 * np.log(h * w / k)
    return {'mx': rng.uniform(0, w, k).astype(np.float32),
            'my': rng.uniform(0, h, k).astype(np.float32),
            'lsx': f(log_std), 'lsy': f(log_std), 'lo': f(0.0),
            'cr': f(0.5), 'cg': f(0.5), 'cb': f(0.5)}


def loss_and_dout(img, target):
    diff = img - target
    return float(np.mean(diff ** 2)), 2 * diff / diff.size


def target_image(path=None, size=None):
    """The target, at its native resolution or with its longer side resized to `size`.
    Default: target.png (256x256), generated on first use."""
    if path is None:
        path = os.path.join(here, 'target.png')
        if not os.path.exists(path):
            # procedural: two discs and a stripe on a gradient background
            y, x = np.mgrid[0:256, 0:256] / 256
            img = np.stack([0.2 + 0.6 * x, 0.3 + 0.4 * y, 0.6 - 0.3 * x], -1)
            img[(x - 0.33) ** 2 + (y - 0.35) ** 2 < 0.04] = (0.95, 0.85, 0.2)
            img[(x - 0.7) ** 2 + (y - 0.65) ** 2 < 0.025] = (0.15, 0.55, 0.9)
            img[np.abs(x + y - 1.1) < 0.04] = (0.9, 0.2, 0.25)
            Image.fromarray((img * 255).astype(np.uint8)).save(path)
    im = Image.open(path).convert('RGB')
    if size:
        s = size / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255


def to_image(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


class Adam:
    def __init__(self, lrs, b1=0.9, b2=0.999, eps=1e-8):
        self.lrs, self.b1, self.b2, self.eps = lrs, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            m = self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
            v = self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            params[k] = (params[k] - self.lrs[k] * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)
