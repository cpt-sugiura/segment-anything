from datetime import datetime
import os
import torch
import cv2
import sys
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add the path of 'segment_anything' module to sys.path
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# MaskGenerator Class
class MaskGenerator:
    # Initialize the class with model type, device, and checkpoint path
    def __init__(self, model_type="vit_h", device="cpu", checkpoint_path=None):
        self.model_type = model_type
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.mask_generator = None

    # Load the model into the specified device
    def load_model(self):
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.model.to(device=self.device)

    # Initialize the mask generator with the given parameters
    def initialize_mask_generator(self, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers,
                                  crop_n_points_downscale_factor, min_mask_region_area):
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area
        )

    # Generate masks, color them, and return them along with their counts
    def generate_and_return_colored_masks(self, image):
        masks = self.mask_generator.generate(image)
        # masks を結果ファイルとして保存
        np.save(f"storage/mask/masks_{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.npy", masks)

        combined_mask = np.zeros_like(image)

        np.random.seed(seed=32)
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            mask = mask.astype(np.uint8)

            random_color = np.random.randint(0, 256, size=(3,))

            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = random_color

            combined_mask += colored_mask
            combined_mask_colored = combined_mask.copy()
            combined_mask_colored[colored_mask > 0] = 0

        combined_mask = np.clip(combined_mask, 0, 255)
        combined_mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2RGB)

        return self.show_anns(image, combined_mask_3ch), combined_mask

    # Display the masks on top of the original image
    def show_anns(self, image, masks):
        fig = plt.figure(figsize=(20, 20))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 0.7, masks, 0.3, 0)

        plt.imshow(image)
        plt.axis('on')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        return img


# Check the existence of the checkpoint file and other specifications
def check_status():
    checkpoint_path = os.path.join("models", "sam_vit_h_4b8939.pth")
    print(checkpoint_path, "; exist:", os.path.isfile(checkpoint_path))
    print("PyTorch version:", torch.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    return checkpoint_path


# Function to process the image and generate masks
def process_image(image, points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers,
                  crop_n_points_downscale_factor, min_mask_region_area):
    # 現在時刻を表示
    print(f"start: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    checkpoint_path = check_status()
    org_image = image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    mask_gen = MaskGenerator(checkpoint_path=checkpoint_path)
    mask_gen.load_model()
    mask_gen.initialize_mask_generator(points_per_side, pred_iou_thresh, stability_score_thresh, crop_n_layers,
                                       crop_n_points_downscale_factor, min_mask_region_area)
    mask_image, combined_mask = mask_gen.generate_and_return_colored_masks(image)
    print(f"  end: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    return org_image, mask_image, combined_mask


# Main function to run the application
if __name__ == "__main__":
    create_directory("images")
    inputs = [
        gr.Image(label="Input Image - Upload an image to be processed."),
        gr.Slider(minimum=4, maximum=64, step=4, value=32,
                  label="Points Per Side - 点のサンプリング密度:探索ステップの増減。増やすと処理時間も増える。減らすと探索範囲が少なくなる"),
        gr.Slider(minimum=0, maximum=1, step=0.001, value=0.980,
                  label="Prediction IOU Threshold - 品質 :値を減らすと検出マスクが増える、増やすと精度の高いマスクが出力される"),
        gr.Slider(minimum=0, maximum=1, step=0.001, value=0.960,
                  label="Stability Score Threshold - 重複マスクの除去の閾値"),
        gr.Slider(minimum=0, maximum=5, step=1, value=0,
                  label="Crop N Layers - 画像の切り抜きに自動的に生成を実行"),
        gr.Slider(minimum=0, maximum=5, step=1, value=2,
                  label="Crop N Points Downscale Factor - 小さなオブジェクトのパフォーマンスを向上させる"),
        gr.Slider(minimum=1, maximum=500, step=1, value=100,
                  label="Min Mask Region Area - 小領域のピクセルや穴を除去できます"),
    ]
    outputs = [
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Overlay Image"),
        gr.Image(type="pil", label="Original mask Image")
    ]

    gr.Interface(fn=process_image, inputs=inputs, outputs=outputs).launch()
