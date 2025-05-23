import argparse

from torchvision.utils import save_image

from utils import util


def run(args):
    if args.model == 'sd3':
        from diffusion.base.sd3_sampler import get_sampler
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    sampler = get_sampler(args.sampler)
    if args.efficient_memory:
        prompt_emb, null_emb = util.precompute_text_embedding(sampler, [args.prompt, args.negative_prompt], device='cuda')
    else:
        prompt_emb = None
        null_emb = None

    sampler = sampler.to(device='cuda')
    output = sampler.sample(prompts=[args.prompt, args.negative_prompt],
                            NFE=args.NFE,
                            img_shape=(args.img_shape, args.img_shape),
                            cfg_scale=args.cfg_scale,
                            prompt_emb=prompt_emb,
                            null_emb=null_emb)

    save_image(output, args.save_path, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sd3', choices=['sd3', 'flux'], help='Model to use for sampling')
    parser.add_argument('--sampler', type=str, default='euler', help='Sampler to use for sampling')
    parser.add_argument('--prompt', type=str, default='A photo of a cat holding "hello world"', help='Prompt to use for sampling')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt to use for sampling')
    parser.add_argument('--img_shape', type=int, default=768, help='Image shape for sampling')
    parser.add_argument('--cfg_scale', type=float, default=7.5, help='CFG scale for sampling')
    parser.add_argument('--save_path', type=str, default='output.png', help='Path to save the output image')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--NFE', type=int, default=28, help='Number of function evaluations for sampling')
    parser.add_argument('--efficient_memory', action='store_true', help='Use efficient memory for sampling')
    args = parser.parse_args()

    util.set_seed(args.seed)
    run(args)