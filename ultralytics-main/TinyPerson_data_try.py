import json
import os
import yaml
from ultralytics import YOLO


def convert_tinyperson_to_yolo(json_path, images_dir, output_labels_dir):
    """
    将 TinyPerson 的 COCO 格式 JSON 转换为 YOLO 格式的 txt 标签
    """
    os.makedirs(output_labels_dir, exist_ok=True)
    print(f"[*] 正在处理标注文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_info = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']} for img in
                   data.get('images', [])}

    category_mapping = {}
    for idx, cat in enumerate(data.get('categories', [])):
        category_mapping[cat['id']] = idx

    yolo_labels = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        cat_id = ann['category_id']

        if cat_id not in category_mapping:
            continue

        yolo_class_id = category_mapping[cat_id]
        x_min, y_min, w, h = ann['bbox']

        img_info = images_info.get(img_id)
        if not img_info:
            continue

        img_width = img_info['width']
        img_height = img_info['height']

        # 中心点和宽高归一化
        x_center = max(0.0, min(1.0, (x_min + w / 2.0) / img_width))
        y_center = max(0.0, min(1.0, (y_min + h / 2.0) / img_height))
        w_norm = max(0.0, min(1.0, w / img_width))
        h_norm = max(0.0, min(1.0, h / img_height))

        line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"

        if img_id not in yolo_labels:
            yolo_labels[img_id] = []
        yolo_labels[img_id].append(line)

    file_count = 0
    for img_id, lines in yolo_labels.items():
        file_name = images_info[img_id]['file_name']
        txt_name = os.path.splitext(os.path.basename(file_name))[0] + '.txt'
        txt_path = os.path.join(output_labels_dir, txt_name)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        file_count += 1

    print(f"[+] 转换完成！共生成了 {file_count} 个标签文件到 {output_labels_dir}\n")


def create_yaml_config(yaml_path, train_dir, val_dir):
    """
    自动生成 YOLO 训练所需的 yaml 配置文件
    """
    config = {
        'train': train_dir,
        'val': val_dir,
        'nc': 2,
        'names': ['earth person', 'sea person']
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"[+] YAML 配置文件已生成: {yaml_path}\n")


def train_yolo11_baseline(yaml_path):
    """
    严格按照论文表1的参数启动 YOLO11 训练
    """
    print("[*] 正在加载 YOLO11 模型并准备训练...")
    # 论文中 YOLO11 基线模型的参数量约为 9.4M，对应 yolo11s.pt 的量级
    model = YOLO('yolo11n.pt')

    # 完全对齐论文的超参数设置
    results = model.train(
        data=yaml_path,
        epochs=2,  # 训练轮次
        imgsz=640,  # 图像尺寸
        batch=8,  # 批样本数量
        workers=4,  # 子进程数量
        optimizer='SGD',  # 优化器
        lr0=0.01,  # 学习率
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        device=0,  # 默认使用第一张 GPU
        project='TinyPerson_Exp',
        name='YOLO11_Baseline'
    )
    print("[+] 训练启动成功！")


if __name__ == '__main__':
    # =====================================================================
    # ⚠️ 请修改这里的路径为你电脑上的实际路径
    # =====================================================================
    BASE_RAW_DIR = r'E:\YOLO\data\OpenDataLab___TinyPerson\raw'  # 指向你刚刚截图的那个 raw 文件夹

    # 请去 annotations 文件夹里确认这两个 json 文件的确切名字，如果不同请修改
    TRAIN_JSON_NAME = 'tiny_set_train.json'
    VAL_JSON_NAME = 'tiny_set_test.json'
    # =====================================================================

    # 自动拼接出所需的各种路径
    train_json_path = os.path.join(BASE_RAW_DIR, 'annotations\\annotations', TRAIN_JSON_NAME)
    val_json_path = os.path.join(BASE_RAW_DIR, 'annotations\\annotations', VAL_JSON_NAME)

    train_images_dir = os.path.join(BASE_RAW_DIR, 'train')
    val_images_dir = os.path.join(BASE_RAW_DIR, 'test')

    # 转换后的标签，YOLO 默认会在图片同级的 labels 文件夹下去找
    train_labels_dir = os.path.join(BASE_RAW_DIR, 'labels', 'train')
    val_labels_dir = os.path.join(BASE_RAW_DIR, 'labels',
                                  'test')  # YOLO 找验证集标签时会根据图片路径替换 'images' 为 'labels'，这里为了方便直接指定

    yaml_config_path = os.path.join(BASE_RAW_DIR, 'tinyperson.yaml')

    # 第一步：转换训练集标签
    if os.path.exists(train_json_path):
        convert_tinyperson_to_yolo(train_json_path, train_images_dir, train_labels_dir)
    else:
        print(f"❌ 找不到训练集标注文件: {train_json_path}")

    # 第二步：转换验证集/测试集标签
    if os.path.exists(val_json_path):
        convert_tinyperson_to_yolo(val_json_path, val_images_dir, val_labels_dir)
    else:
        print(f"❌ 找不到测试集标注文件: {val_json_path}")

    # 注意：YOLO 默认要求图片在 images 目录下，标签在 labels 目录下。
    # 如果 TinyPerson 的图片直接放在 train 目录下，我们需要通过 yaml 指定明确的绝对路径
    # 第三步：生成 YAML 配置文件
    create_yaml_config(yaml_config_path, train_images_dir, val_images_dir)

    # 第四步：启动基线模型训练
    train_yolo11_baseline(yaml_config_path)