# SwinURNet: Hybrid Transformer-CNN Architecture for Real-time Unstructured Road Segmentation

Code for our paper:
> **SwinURNet: Hybrid Transformer-CNN Architecture for Real-time Unstructured Road Segmentation**
> <br>Zhangyu Wang, Zhihao Liao, Guizhen Yu*, Bin Zhou, Wenwen Luo<br>


## Abstract：
Semantic segmentation is a crucial component for autonomous driving. However, the segmentation performance in unstructured road is challenging owing to the following reasons：(1) irregular shapes and varying sizes of road boundaries, (2) low contrast or blurred boundaries between the road and background, and (3) environmental factors such as changing light intensities and dust particles. To overcome these challenges, this study proposes SwinURNet, a Transformer-convoluted neural network (CNN) architecture for real-time point cloud segmentation in unstructured scenarios. First, the point cloud is projected onto a range image via spherical projection. Then, a lightweight ResNet34-based network is designed for encoding abstract features. A non-square Swin transformer is designed for decoding information and capturing high-resolution transverse features. A multidimensional information fusion module is introduced to balance the semantic differences between features maps in CNNs and attention maps in transformers. A multitask loss function comprising boundary, weighted cross-entropy, and Lovász–Softmax losses is introduced at the network end to guide network training. Experimental data from autonomous mining trucks in the Baiyuneboite mining area is used to evaluate the performance on unstructured roads. The proposed architecture is also applied on the public unstructured dataset RELLIS-3D and the large structured dataset SemanticKITTI. The experimental results show performance gains of 74.2, 42.6, and 61.6% in accuracy and 8–19 FPS inference speed with the proposed architecture, surpassing those of the compared methods.
<p align="center">
   <img src="assert/1.png" width="80%"> 
</p>

**2024-06-21[:yum:]** Release training and testing codes of SwinURNet. You can find more details in the code.

**2024-06-20[:sunglasses:]** Release Mining site dataset, it is an unstructured dataset collected by LiDAR deployed on autonomous mining trucks in the Baiyuneboite, Inner Mongolia, China. 

## Prepare:
Download Mining site dataset from [baidu cloud disk](https://pan.baidu.com/s/1gJtstG1oNJJnnEuO_C7nFQ?pwd=1234). Download RELLIS-3D dataset from [baidu cloud disk](https://pan.baidu.com/s/1wWoqMd-aE7OwPgP3tWkeyg?pwd=1234). Download SemanticKITTI dataset from [official web](http://www.semantic-kitti.org/dataset.html).

## Usage：
### Train：
- SemanticKITTI:

    `python train.py -d /your_dataset -ac config/arch/swinurnet-512.yml -n swinurnet-512`

    Note that the following training strategy is used due to GPU and time constraints, see [kitti.sh] for details.

    First train the model with 64x512 inputs. Then load the pre-trained model to train the model with 64x1024 inputs.
    
    Also, for this reason, if you want to resume training from a breakpoint, and change "/SwinURNet_valid_best" to "/SwinURNet".



### Infer and Eval：
- SemanticKITTI:

    `python infer.py -d /your_dataset -l /your_predictions_path -m trained_model -s valid/test`
    
    Eval for valid sequences:

    `python evaluate_iou.py -d /your_dataset -p /your_predictions_path`

    For test  sequences, need to upload to [CodaLab](https://competitions.codalab.org/competitions/20331#participate) pages.


### Visualize Example:


- Visualize GT:

  `python visualize.py -w kitti -d /your_dataset -s what_sequences`

- Visualize Predictions:

  `python visualize.py -w kitti -d /your_dataset -p /your_predictions -s what_sequences`



## Acknowledgments：
Code framework derived from [SalsaNext](https://github.com/Halmstad-University/SalsaNext). Part of code from [Swin Transformer](https://github.com/microsoft/Swin-Transformer). Thanks to their open source code.


