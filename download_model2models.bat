@echo off

REM ディレクトリ "models" を作成（すでに存在する場合はスキップ）
if not exist "models" mkdir models

REM 各モデルファイルをダウンロード
echo Downloading ViT-H SAM model...
curl -L -o models\sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo Downloading ViT-L SAM model...
curl -L -o models\sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

echo Downloading ViT-B SAM model...
curl -L -o models\sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo All downloads completed.
