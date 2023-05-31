import albumentations as A
# 对示例数据集进行增强, 运行成功后会在相应目录下保存
import os
import json
import cv2
import numpy as np


# 定义类
class YOLOAug(object):

    def __init__(self,
                 pre_image_path=None,
                 pre_label_path=None,
                 aug_save_image_path=None,
                 aug_save_label_path=None,
                 labels=None,
                 is_show=True,
                 start_filename_id=None,
                 max_len=4):
        """

        :param pre_image_path:
        :param pre_label_path:
        :param aug_save_image_path:
        :param aug_save_label_path:
        :param labels: 标签列表, 需要根据自己的设定, 用于展示图片
        :param is_show:
        :param start_filename_id:
        :param max_len:
        """
        self.pre_image_path = pre_image_path
        self.pre_label_path = pre_label_path
        self.aug_save_image_path = aug_save_image_path
        self.aug_save_label_path = aug_save_label_path
        self.labels = labels
        self.is_show = is_show
        self.start_filename_id = start_filename_id
        self.max_len = max_len
        # 数据增强选项
        # self.aug = A.Compose([
        #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
        #     A.GaussianBlur(p=0.7),
        #     A.GaussNoise(p=0.7),
        #     A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.5),  # 直方图均衡
        #     A.Equalize(p=0.5),  # 均衡图像直方图
             #A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5)

        #     A.OneOf([
        #         # A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
        #         # A.ChannelShuffle(p=0.3),  # 随机排列通道
        #         # A.ColorJitter(p=0.3),  # 随机改变图像的亮度、对比度、饱和度、色调
        #         # A.ChannelDropout(p=0.3),  # 随机丢弃通道
        #     ], p=0.),
        #     # A.Downscale(p=0.1),  # 随机缩小和放大来降低图像质量
        #     A.Emboss(p=0.2),  # 压印输入图像并将结果与原始图像叠加
        # ],
        self.aug = A.Compose([
            #A.GaussNoise( p=1.0)
            #A.VerticalFlip(always_apply=True, p=1.0)
            #A.HorizontalFlip(always_apply=True, p=1.0)
            A.RandomBrightness(limit=0.7, p=1.0),
            #A.MotionBlur(blur_limit=7, always_apply=True, p=1.0),
            #A.RandomResizedCrop(1586,3034,(0.6,0.6),p=1.0)
            #A.ShiftScaleRotate(border_mode=0, value=[255, 255, 255],p=1.0)
            # 第一次
            # A.GlassBlur(sigma=0.7, max_delta=4, p=0.3),
            # A.Rotate(limit=89, p=0.8),

            # A.ISONoise(color_shift=(0.01,0.05),intensity=(0.5,0.8),always_apply=False,p=1.0),

            # 第二次
            # A.Downscale(p=0.3),  # 随机缩小和放大来降低图像质量
            # A.ColorJitter(p=0.5),  # 随机改变图像的亮度、对比度、饱和度、色调
            # A.ChannelShuffle(p=0.3),  # 随机排列通道
            #A.Emboss(p=0.2),  # 压印输入图像并将结果与原始图像叠加
            #A.Equalize(p=0.5),  # 均衡图像直方图

            #A.RandomSnow(p=1), # 加雪花
            #A.RandomRain(p=1), # 加雨滴
            #A.RandomFog(fog_coef_lower=0.3,fog_coef_upper=0.5,p=1), # 加雾
            #A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5,p=1), # 加阳光
        ],
            # yolo: [x_center, y_center, width, height]  # 经过归一化
            # min_area: 表示bbox占据的像素总个数, 当数据增强后, 若bbox小于这个值则从返回的bbox列表删除该bbox.
            # min_visibility: 值域为[0,1], 如果增强后的bbox面积和增强前的bbox面积比值小于该值, 则删除该bbox
            A.BboxParams(format='yolo', min_area=0., min_visibility=0., label_fields=['category_id'])

        )
        print("--------*--------")
        image_len = len(os.listdir(self.pre_image_path))
        print("the length of images: ", image_len)
        if self.start_filename_id is None:
            print("the start_filename id is not set, default: len(image)", image_len)
            self.start_filename_id = image_len

        print("--------*--------")

    def get_data(self, image_name):
        """
        获取图片和对应的label信息

        :param image_name: 图片文件名, e.g. 0000.jpg
        :return:
        """
        image = cv2.imread(os.path.join(self.pre_image_path, image_name))

        if len(image_name.split('.')[0]) == 0:
            return None

        with open(os.path.join(self.pre_label_path, image_name.split('.')[0] + '.txt'), 'r', encoding='utf-8') as f:
            label_txt = f.readlines()

        label_list = []
        cls_id_list = []
        for label in label_txt:
            label_info = label.strip().split(' ')
            cls_id_list.append(int(label_info[0]))
            label_list.append([float(x) for x in label_info[1:]])

        anno_info = {'image': image, 'bboxes': label_list, 'category_id': cls_id_list}
        return anno_info

    def aug_image(self):
        image_list = os.listdir(self.pre_image_path)

        file_name_id = self.start_filename_id
        for image_filename in image_list[:]:
            image_suffix = image_filename.split('.')[-1]
            if image_suffix not in ['jpg', 'png']:
                continue
            image_suffix = image_filename.split('.')[-1]

            aug_anno = self.get_data(image_filename)
            if aug_anno is None:
                continue

            # 获取增强后的信息
            augmented = self.aug(**aug_anno)  # {'image': , 'bboxes': , 'category_id': }

            # aug_image = aug_info['image']
            # aug_bboxes = aug_info['bboxes']
            # aug_category_id = aug_info['category_id']
            # change
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_category_id = augmented['category_id']

            name = '0' * self.max_len
            cnt_str = str(file_name_id)
            length = len(cnt_str)
            new_image_filename = name[:-length] + cnt_str + f'.{image_suffix}'
            new_label_filename = name[:-length] + cnt_str + '.txt'
            print(f"aug_image_{new_image_filename}: ")

            aug_image_copy = aug_image.copy()
            for cls_id, bbox in zip(aug_category_id, aug_bboxes):
                print(f" --- --- cls_id: ", cls_id)

                if self.is_show:
                    tl = 2
                    h, w = aug_image_copy.shape[:2]
                    x_center = int(bbox[0] * w)
                    y_center = int(bbox[1] * h)
                    width = int(bbox[2] * w)
                    height = int(bbox[3] * h)
                    xmin = int(x_center - width / 2)
                    ymin = int(y_center - height / 2)
                    xmax = int(x_center + width / 2)
                    ymax = int(y_center + height / 2)
                    text = f"{self.labels[cls_id]}"
                    t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tl)[0]
                    cv2.rectangle(aug_image_copy, (xmin, ymin - 3), (xmin + t_size[0], ymin - t_size[1] - 3),
                                  (0, 0, 255),
                                  -1, cv2.LINE_AA)  # filled
                    cv2.putText(aug_image_copy, text, (xmin, ymin - 2), 0, tl / 3, (255, 255, 255), tl, cv2.LINE_AA)
                    aug_image_show = cv2.rectangle(aug_image_copy, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

            if self.is_show:
                cv2.imshow(f'aug_image_{new_image_filename}', aug_image_show)
                key = cv2.waitKey(0)
                # 按下s键保存增强，否则取消保存此次增强
                if key & 0xff == ord('s'):
                    pass
                else:
                    cv2.destroyWindow(f'aug_image_{new_image_filename}')
                    continue
                cv2.destroyWindow(f'aug_image_{new_image_filename}')

            # 保存增强后的信息
            cv2.imwrite(os.path.join(self.aug_save_image_path, new_image_filename), aug_image)
            with open(os.path.join(self.aug_save_label_path, new_label_filename), 'w', encoding='utf-8') as lf:
                for cls_id, bbox in zip(aug_category_id, aug_bboxes):
                    lf.write(str(cls_id) + ' ')
                    for i in bbox:
                        # 保存小数点后六位
                        lf.write(str(i)[:8] + ' ')
                    lf.write('\n')

            file_name_id += 1


# 原始图片和label路径

PRE_IMAGE_PATH =  r'C:\Users\xkw\Desktop\pcb-box\test1/'
PRE_LABEL_PATH =  r'C:\Users\xkw\Desktop\pcb-box\test2/'

# 增强后的图片和label保存的路径
AUG_SAVE_IMAGE_PATH = r'C:\Users\xkw\Desktop\pcb-box/'
AUG_SAVE_LABEL_PATH = r'C:\Users\xkw\Desktop\pcb-box/'

# 类别列表, 需要根据自己的修改
labels = ['crazing', 'inclusion', 'patches','pitted_surface','rolled-in_scale','scratches']


aug = YOLOAug(pre_image_path=PRE_IMAGE_PATH,
                pre_label_path=PRE_LABEL_PATH,
                aug_save_image_path=AUG_SAVE_IMAGE_PATH,
                aug_save_label_path=AUG_SAVE_LABEL_PATH,
                labels=labels,
                is_show=False)
aug.aug_image()

