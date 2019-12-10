# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import argparse
import cv2, os

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

from glob import glob


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/centermask/centermask_M_v2_FPN_lite_res600_ms_bs32_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="centermask-lite-M-v2-ms-bs32-1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--conf_th",
        type=float,
        default=0.2,
        help="confidence_threshold",
    )
    parser.add_argument(
        '--display_text',
        default=False,
        type=str2bool,
        help='Whether or not to display text (class [score])')
    parser.add_argument(
        '--display_scores',
        default=False,
        type=str2bool,
        help='Whether or not to display scores in addition to classes')
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/coco/test2017',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        default='demo/results/CenterMask-Lite-M-v2',
        help='directory to save demo results'
        )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.

    # demo_im_names = os.listdir(args.images_dir)

    # # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.conf_th,
        display_text = args.display_text,
        display_scores = args.display_scores
    )

    # for im_name in demo_im_names:
    #     img = cv2.imread(os.path.join(args.images_dir, im_name))
    #     if img is None:
    #         continue
    #     start_time = time.time()
    #     composite = coco_demo.run_on_opencv_image(img)
    #     print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    #     cv2.imshow(im_name, composite)
    # print("Press any keys to exit ...")
    # cv2.waitKey()
    # cv2.destroyAllWindows()



    if os.path.isfile(args.input):
        imglist = [args.input]
    else:
        imglist = glob(os.path.join(args.input, '*'))

    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(num_images):
        print('file', i)
        if os.path.splitext(imglist[i])[1] in ['.mp4', '.wmv', '.avi']:
            cap = cv2.VideoCapture(imglist[i])
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # scale = get_target_scale(image_width, image_height, 800, 1333)
            video = cv2.VideoWriter(os.path.join(args.output_dir, os.path.basename(imglist[i])),
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    25.0,
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                                    )
            cnt = 0
            while True:
                cnt += 1
                # Capture frame-by-frame
                ret, img = cap.read()

                if ret:
                    start_time = time.time()
                    composite = coco_demo.run_on_opencv_image(img)
                    print("{} frame \tinference time: {:.2f}s".format(cnt, time.time() - start_time))                    
                    video.write(composite)
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            video.release()
            print('Result file number {} saved'.format(i))
        elif os.path.splitext(imglist[i])[1] in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']:
            img = cv2.imread(imglist[i])
            assert img is not None
            im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
            print(f"{im_name} processing...")
            start_time = time.time()
            composite = coco_demo.run_on_opencv_image(img)
            print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
            save_path = os.path.join(args.output_dir, f'{im_name}_result.jpg')
            cv2.imwrite(save_path, composite)

        else:
            continue







if __name__ == "__main__":
    main()

