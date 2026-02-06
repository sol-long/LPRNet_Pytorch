from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

# ================= 字符表配置 =================
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-']

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        
        # 支持传入列表或单个路径字符串
        if isinstance(img_dir, str):
            img_dir = [img_dir]
            
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
            
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        
        # --- 1. 智能解析车牌号 ---
        basename = os.path.basename(filename)
        root_name = os.path.splitext(basename)[0]
        
        # 2. 按下划线切割
        parts = root_name.split('_')
        
        # 3. 寻找真正的车牌 (包含省份字符的那一段)
        label_str = ""
        for part in parts:
            if len(part) > 1 and part[0] in CHARS[:31]: 
                label_str = part
                break
        
        # 兜底逻辑：如果没找到，且文件名本身就是车牌开头
        if label_str == "":
            if root_name[0] in CHARS[:31]:
                label_str = root_name
            else:
                # print(f"⚠️ 无法解析车牌: {filename}")
                return self.__getitem__(0)

        # --- 2. 图片读取与预处理 ---
        try:
            Image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
            height, width, _ = Image.shape
        except Exception:
            return self.__getitem__(0)

        # 尺寸调整
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, (self.img_size[0], self.img_size[1]))
        Image = self.PreprocFun(Image)

        # --- 3. 标签转换 ---
        label = list()
        for c in label_str:
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            else:
                # 遇到非法字符直接跳过，防止报错
                pass

        if len(label) == 0:
             return self.__getitem__(0)

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        return True