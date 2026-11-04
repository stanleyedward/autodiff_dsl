# fit.py  (python apps/splat2d/fit.py [--target img.png] [--size 512] [--k 1024] [--save-frames 10])
# Fits the target (at its native resolution) with K Gaussians by Adam.
# Writes results/fit.gif (first config; frames at log-spaced steps, titled with the loss)
# and, with --save-frames N, every N-th render to results/frames/<config>/step_XXXX.png.
import argparse
from PIL import ImageDraw, ImageFont
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--target', help='image to fit (default: target.png, 256x256)')
ap.add_argument('--size', type=int, help='resize so the longer side is SIZE px (default: native)')
ap.add_argument('--k', type=int, default=1024, help='number of Gaussians')
ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--configs', nargs='+', default=['all'], choices=list(CONFIGS))
ap.add_argument('--save-frames', type=int, default=0, metavar='N',
                help='save the render every N steps to results/frames/<config>/')
args = ap.parse_args()

LRS = {'mx': 0.5, 'my': 0.5, 'lsx': 0.01, 'lsy': 0.01, 'lo': 0.05, 'cr': 0.05, 'cg': 0.05, 'cb': 0.05}
# most of the change happens in the first ~100 steps, so the GIF samples steps log-spaced
GIF_STEPS = set(np.unique(np.round(np.geomspace(1, args.steps, 48))).astype(int)) | {0}
results_dir = os.path.join(here, 'results')
os.makedirs(results_dir, exist_ok=True)

target = target_image(args.target, args.size)
h, w = target.shape[:2]
inputs = pixel_coords(h, w)
init = init_params(args.k, h, w)
print(f'{w}x{h} target, {args.k} Gaussians')


def gif_frame(img, step, loss):
    dw = min(max(w, 256), 512)
    im = to_image(img).resize((dw, round(h * dw / w)))
    frame = Image.new('RGB', (im.width, im.height + 24), 'white')
    frame.paste(im, (0, 24))
    ImageDraw.Draw(frame).text((6, 4), f'step {step}   loss {loss:.4f}', fill='black',
                               font=ImageFont.load_default(size=14))
    return frame


for i, config in enumerate(args.configs):
    r = Renderer(config, args.k)
    params = {k: v.copy() for k, v in init.items()}
    adam = Adam(LRS)
    frames_dir = os.path.join(results_dir, 'frames', config.replace(' ', '').replace('+', '_'))
    if args.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    losses, gif = [], []
    for step in range(args.steps + 1):
        img = r.render(inputs, params)
        loss, dout = loss_and_dout(img, target)
        losses.append(loss)
        if i == 0 and step in GIF_STEPS:
            gif.append(gif_frame(img, step, loss))
        if args.save_frames and step % args.save_frames == 0:
            to_image(img).save(os.path.join(frames_dir, f'step_{step:04d}.png'))
        if step == args.steps:
            break
        adam.step(params, r.grads(inputs, params, dout))
    print(f'{config}: loss {losses[0]:.5f} -> {losses[-1]:.5f}')
    if gif:
        gif[0].save(os.path.join(results_dir, 'fit.gif'), save_all=True, append_images=gif[1:],
                    duration=[120] * (len(gif) - 1) + [2000], loop=0)
