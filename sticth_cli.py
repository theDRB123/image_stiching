import argparse from pano_core import Panaroma from helpers import load_and_resize, show_and_save def main(): ap = argparse.ArgumentParser() ap.add_argument("images", nargs="+", help="image paths in left→right order") ap.add_argument("-w", "--width",  type=int, default=400) ap.add_argument("-H", "--height", type=int, default=400)  # use -H to avoid -h conflict
    args = ap.parse_args()

    imgs = [load_and_resize(p, args.width, args.height) for p in args.images]

    pano = Panaroma()
    if len(imgs) == 2:
        res, vis = pano.image_stitch([imgs[0], imgs[1]], match_status=True)
    else:
        res, vis = pano.image_stitch([imgs[-2], imgs[-1]], match_status=True)
        for i in range(len(imgs) - 2):
            res = pano.image_stitch([imgs[-i-3], res])[0]

    show_and_save(res, vis)
if __name__ == "__main__":
    main()
