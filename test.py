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
from typing import Any, Dict, List, Tuple
from plyer import notification

from app.get_angle import get_angle

# Add the path of 'segment_anything' module to sys.path
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_rectangle_similarity(mask, rect):
    """
    マスク領域を囲む長方形がマスクに非常に似た形であるかを評価する関数
    """
    mask_area = np.sum(mask)
    rect_area = cv2.contourArea(cv2.boxPoints(rect))

    similarity = min(mask_area, rect_area) / max(mask_area, rect_area)
    return similarity > 0.9


def get_solar_panel_masks_by_filter(segment_masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    次の条件に合うようにオブジェクトが1それ以外が0になっている segment_masks を絞り込んでソーラーパネルのマスクのみにする
        - 長方形である
        - 他のソーラーパネルと並列に配置されている
    """
    rectangles: List[List[Tuple[np.ndarray, cv2.RotatedRect]]] = [[] for _ in range(len(segment_masks))]
    has_rectangle_masks: List[bool] = [False] * len(segment_masks)

    print('長方形を持つマスクを識別します')
    # ステップ1: 長方形の識別
    for i, mask_data in enumerate(segment_masks):
        mask = mask_data['segmentation']
        mask = mask.astype(np.uint8)
        # 外部輪郭を抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # # 輪郭を近似するポリゴンを取得
            # epsilon = 0.05 * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            # # 近似したポリゴンの頂点数が4つであり、それぞれが90度の角を成す場合、長方形とみなします
            # if len(approx) < 4:
            #     continue
            # # 角度チェックの柔軟化
            # angles = []
            # for j in range(len(approx)):
            #     # (0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1) の順に回しています
            #     angle = get_angle(approx[j], approx[(j + 1) % len(approx)], approx[(j + 2) % len(approx)])
            #     angles.append(angle)
            # ninety_angles = list(filter(lambda x: 80 <= x <= 100, angles))  # 90度に近いかチェック
            # if len(ninety_angles) == 4:  # 90度に近い角が4つなら長方形に近いとする
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            # rectangles[i].append((box, rect))
            # has_rectangle_masks[i] = True
            if check_rectangle_similarity(mask, rect):
                rectangles[i].append((box, rect))
                has_rectangle_masks[i] = True
    print('長方形を持つマスクを識別しました')
    print(has_rectangle_masks)
    # 長方形を持つ mask だけ返す
    filtered_masks = [mask for i, mask in enumerate(segment_masks) if has_rectangle_masks[i]]  # 並列配置のチェックもここで行う
    return filtered_masks

    solar_panel_masks = []

    # ステップ2: 長方形の関係性の分析
    for i, (_, rect1) in enumerate(rectangles):
        parallel_and_same_direction = True

        for j, (_, rect2) in enumerate(rectangles):
            if i == j:
                continue

            if not is_parallel_and_same_direction(rect1, rect2):
                parallel_and_same_direction = False
                break

        if parallel_and_same_direction:
            # ソーラーパネルとして追加
            solar_panel_masks.append(segment_masks[i])

    return solar_panel_masks


def is_parallel_and_same_direction(rect1, rect2):
    # rect1とrect2が並列かつ同じ方向を向いているかを判断するロジック
    # ここに方向と角度を比較するコードを実装

    # 例: 角度の差が小さいかどうかで判断
    angle_diff = abs(rect1[2] - rect2[2])
    return angle_diff < 10  # 10度以内の差を許容


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
        try:
            masks: List[Dict[str, Any]] = self.mask_generator.generate(image)
            mask_points = []
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

            combined_mask = np.clip(combined_mask, 0, 255)
            combined_mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2RGB)

            # ソーラーパネルのマスクを抽出して描画
            solar_panel_masks = get_solar_panel_masks_by_filter(masks)
            all_solar_panel_mask = np.zeros_like(image)
            green_color = np.array([0, 255, 0])
            for i, mask_data in enumerate(solar_panel_masks):
                mask = mask_data['segmentation']
                mask = mask.astype(np.uint8)
                # カラーマスクを適用
                colored_mask = np.zeros_like(image)
                colored_mask[mask == 1] = green_color
                all_solar_panel_mask += colored_mask
            all_solar_panel_mask = np.clip(all_solar_panel_mask, 0, 255)

            notification.notify(
                title='処理完了',
                message='画像の処理が完了しました。',
                app_name='Image Processor'
            )

            return self.show_anns(image, combined_mask_3ch), combined_mask, all_solar_panel_mask
        except Exception as e:
            # エラー通知
            notification.notify(
                title='処理失敗',
                message='画像処理中にエラーが発生しました。',
                app_name='Image Processor'
            )
            raise e

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
    mask_image, combined_mask, solar_panel_mask = mask_gen.generate_and_return_colored_masks(image)
    print(f"  end: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    return org_image, mask_image, combined_mask, solar_panel_mask


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
        gr.Image(type="pil", label="Original mask Image"),
        gr.Image(type="pil", label="solar_panel_mask")
    ]

    gr.Interface(fn=process_image, inputs=inputs, outputs=outputs).launch()
