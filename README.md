Global Contextual Attention Augmented YOLO with ConvMixer Prediction Heads for PCB Surface Defect Detection
=
1.GCC-YOLO
----
To solve the problem of missed and false detection caused by the large number of tiny targets and complex background textures in a printed circuit board (PCB), we propose a global contextual attention augmented YOLO model with ConvMixer prediction heads (GCC-YOLO). In this study, we apply a high-resolution feature layer (P2) to gain more details and positional information of small targets. Moreover, in order to suppress the background noisy information and further enhance the feature extraction capability, a global contextual attention module (GC) is introduced in the backbone network and combined with a C3 module. Furthermore, in order to reduce the loss of shallow feature information due to the deepening of network layers, a bi-directional weighted feature pyramid (BiFPN) feature fusion structure is introduced. Finally, a ConvMixer module is introduced and combined with the C3 module to create a new prediction head, which improves the small target detection capability of the model while reducing the parameters. Test results on the PCB dataset show that GCC-YOLO improved the Precision, Recall, mAP@0.5, and mAP@0.5:0.95 by 0.2%, 1.8%, 0.5%, and 8.3%, respectively, compared to YOLOv5s; moreover, it has a smaller model volume and faster reasoning speed compared to other algorithms.  <br>  <br>
2.Improved Network Architecture
----
![image](https://github.com/2462954048/GCC-YOLO-V2/assets/45593319/2d8fd4d8-20a4-4692-8e82-bd3a685440cd)  <br>  <br>
We have published the code for the proposed C3GC module and C3CM module, which can be accessed at the link: (https://github.com/2462954048/GCC-YOLO-V2/tree/master/models/gc.py), (https://github.com/2462954048/GCC-YOLO-V2/tree/master/models/cspcm.py).  <br>  <br>
3.Implementation
----
The proposed architecture is implemented using the PyTorch framework (1.8.1+cu111) with a single GeForce RTX 3090 GPU of 24 GB memory.  <br>  <br>
3.1 Dataset
-----
This experiment is based on the PCB defect dataset released by the Open Lab of Peking University for image augmentation. The original dataset contained 693 images, and the augmented dataset has been expanded to 4158 images.   <br>
We have published the augmented PCB dataset, which can be accessed at the link: (https://pan.baidu.com/s/1kIeh9JCOBKF39E26A-UsBQ), and the download code: 0000.  <br>  <br>
We have published the code of PCB dataset enhancement based on the albumentations framework, which can be accessed at the link: (https://github.com/2462954048/GCC-YOLO-V2/tree/master/AlbumentationEnhance.py). <br>  <br>
3.2 Trained model
-----
You can download the weight file from the following links: (https://github.com/2462954048/GCC-YOLO-V2/tree/master/GCC-YOLO.pt)  <br>  <br>
4.License
----
The source code is free for research and education use only. Any comercial use should get formal permission first.  <br>  <br>
5.Contact
----
Please contact 2021203258@cqust.edu.cn for any further questions.



