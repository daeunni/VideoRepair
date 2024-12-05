
from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, SemanticSAMPredictor
import numpy as np 
import warnings;warnings.filterwarnings('ignore')
import cv2, os
import argparse
import ast


def point2mask_semanticsam(img_path, point_lists, img_width=512, img_height=320) : 
    all_point_masks = []

    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    mask_generator = SemanticSAMPredictor(model = build_semantic_sam(model_type='L',    # 'L' / 'T'
                                                                    ckpt=os.path.join(current_file_dir,'checkpoint/swinl_only_sam_many2many.pth'), 
                                                                    current_file_dir = current_file_dir), 
                                                                    thresh=0.5) 
                                                                    
                                          
    for point in point_lists:     # point is pixel point 
        relative_point = point / np.array([img_width, img_height])
        relative_point = [relative_point.tolist()]

        original_image, input_image = prepare_image(image_pth=img_path)  # change the image path to your image
        mask_array, iou_sort_masks, area_sort_masks = mask_generator.predict_masks(original_image, input_image, point=relative_point) # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
        top_iou_mask = mask_array[0]

        # resize 
        if top_iou_mask.shape != (img_height, img_width) : 
            top_iou_mask = cv2.resize(top_iou_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        all_point_masks.append(top_iou_mask)
    
    return np.stack(all_point_masks, axis=0)          # (# of point, 320, 512)

if __name__=='__main__' : 
    parser = argparse.ArgumentParser(description="Semantic SAM")
    parser.add_argument("--img_path", type=str, help='image path', default='first_frame.jpg')
    parser.add_argument("--point_lists", type=str, help='point lists', default='[[184.32, 137.60], [404.48, 169.60]]')
    parser.add_argument("--mask_save_path", type=str, help='mask save path', default='ssam_mask_stack.npy')
    args = parser.parse_args()

    # point list 
    pointlists = ast.literal_eval(args.point_lists)    # string -> list (e.g., [[184.32, 137.60], [404.48, 169.60]])
    assert isinstance(pointlists, list), "The variable is not a list!"

    mask_stack = point2mask_semanticsam(img_path=args.img_path, point_lists=pointlists)          # (# of point, 320, 512) numpy array 
    np.save(args.mask_save_path, mask_stack)
    print('* Semantic-SAM mask is saved in here: ', args.mask_save_path)

