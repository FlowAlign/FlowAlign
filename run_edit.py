import argparse
from pathlib import Path

from torchvision.utils import save_image

from utils import util


def run(args):
    # prepare workdir
    workdir = Path(args.workdir)
    source_dir = workdir.joinpath('source')
    result_dir = workdir.joinpath('edited')
    source_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # load source image
    src_img = util.load_img(args.img_path, img_size=(args.img_shape, args.img_shape))
    src_img = src_img * 2.0 - 1.0

    # prepare method
    if args.model == 'sd3':
        from diffusion.editing.sd3_edit import get_editor
    else:
        raise ValueError(f"Unknown model: {args.model}")
    sampler = get_editor(args.method)

    # pre-compute text embeddings (if efficient_memoery=True)
    if args.efficient_memory:
        src_prompt_emb, tgt_prompt_emb, null_prompt_emb = \
            util.precompute_text_embedding(sampler, 
                                           [args.src_prompt, args.tgt_prompt, args.null_prompt],
                                           device='cuda')
    else:
        src_prompt_emb = None
        tgt_prompt_emb = None
        null_prompt_emb = None

    # DO editing
    sampler = sampler.to(device='cuda')
    output = sampler.sample(src_img=src_img,
                            src_prompt=args.src_prompt,
                            tgt_prompt=args.tgt_prompt,
                            null_prompt="",
                            NFE=args.NFE,
                            img_shape=(args.img_shape, args.img_shape),
                            n_start=args.n_start,
                            cfg_scale=args.cfg_scale,
                            src_prompt_emb=src_prompt_emb,
                            tgt_prompt_emb=tgt_prompt_emb,
                            null_prompt_emb=null_prompt_emb)

    save_image(output, result_dir.joinpath(Path(args.img_path).name), normalize=True)
    save_image(src_img, source_dir.joinpath(Path(args.img_path).name), normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sd3', help='Model to use for sampling')
    parser.add_argument('--method', type=str, default='flowalign', help='Method for editing')
    parser.add_argument('--img_path', type=str, default='samples/cat.jpg')
    parser.add_argument('--src_prompt', type=str, default="a photo of a closed face of a cat")
    parser.add_argument('--tgt_prompt', type=str, default="a photo of a closed face of a dog")
    parser.add_argument('--img_shape', type=int, default=768, help='Image shape for editing')
    parser.add_argument('--cfg_scale', type=float, default=10.0, help='CFG scale for editing')
    parser.add_argument('--workdir', type=str, default='workdir/', help='Directory to save the output image')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for editing')
    parser.add_argument('--NFE', type=int, default=33, help='Number of function evaluations for editing')
    parser.add_argument('--n_start', type=int, default=0, help='Start timestep for editing')
    parser.add_argument('--efficient_memory', action='store_true', help='Use efficient memory for editing')
    args = parser.parse_args()

    util.set_seed(args.seed)
    run(args)
