import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET


def parse_voc_xml(xml_file):
    """解析VOC格式的XML文件"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels


def show_voc_annotation(image_path, annotation_path):
    """显示VOC图像及其标注"""
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 解析标注
    boxes, labels = parse_voc_xml(annotation_path)

    # 创建绘图
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # 绘制边界框和标签
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # 绘制矩形框
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)

        # 添加标签
        plt.text(
            xmin, ymin - 5, label,
            color='white', bbox=dict(facecolor='red', alpha=0.7)
        )

    plt.axis('off')
    plt.title(os.path.basename(image_path))
    plt.show()


# 使用示例
voc_root = 'VOCdevkit/VOC2007'
image_id = '009949'  # 示例图像ID

image_path = os.path.join(voc_root, 'JPEGImages', f'{image_id}.jpg')
annotation_path = os.path.join(voc_root, 'Annotations', f'{image_id}.xml')

show_voc_annotation(image_path, annotation_path)