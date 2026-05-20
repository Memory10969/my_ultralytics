import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.nn.modules.AddModules.LogWaveletDenoise import LogWaveletDenoise  # 根据实际路径调整


def denoise_and_compare(
    input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
    output_dir=r"./denoised_comparison_fixed",
    num_images=50,
    img_size=640,
    threshold_factor=0.3
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 加载修正后的去噪模块
    denoiser = LogWaveletDenoise(level=1, threshold_factor=threshold_factor).to(device)
    denoiser.eval()

    all_imgs = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    selected = sorted(all_imgs)[:num_images]

    print(f"Processing {len(selected)} images with threshold_factor={threshold_factor}...")
    for idx, img_name in enumerate(selected):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            denoised_tensor = denoiser(img_tensor)

        denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        denoised_np = np.clip(denoised_np, 0, 1)

        # 保存对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(denoised_np)
        axes[1].set_title("Denoised Image (Orthogonal Haar)")
        axes[1].axis('off')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(selected)} images")

    print("Done. Output saved to", output_dir)


if __name__ == "__main__":
    # # 首先验证可逆性：threshold_factor=0 应输出原图
    # print("Step 1: Verification with threshold_factor=0 ...")
    # denoise_and_compare(
    #     input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
    #     output_dir=r"./denoised_verify",
    #     num_images=5,
    #     threshold_factor=0.0
    # )

    # 验证通过后，执行真实去噪
    print("\nStep 2: Real denoising with threshold_factor=0.3 ...")
    denoise_and_compare(
        input_dir=r"D:\Study\PostGraduate\YOLO_ultralytics\ultralytics\wbb\datasets\HRSID_YOLO\images\train",
        output_dir=r"./denoised_comparison_fixed",
        num_images=50,
        threshold_factor=1.0
    )