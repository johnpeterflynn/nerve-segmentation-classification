## Multitask Learning on Nerve Ultrasound Images

This repository contains several multi-task extensions of a U-Net model[1] to improve segmentation results on a small ultrasound nerve dataset. Our approaches were guided by TUM's chair for Computer Aided Medical Procedures. We applied multi-task learning[2] with a U-Net model to improve segmentation results on a very limited dataset. We implemented multiple architectures including hard parameter sharing using an FCN classifier at the U-net bottleneck, soft parameter sharing using cross-stitch networks[3] as well as a ResNet-18 benchmark classifier. Our classifiers used cross entropy loss and segmenters used dice loss. We experimented with several multitask loss approaches including linear weighting of classification and segmentation loss, uncertainty weighting[4] and loss scheduling[5].

## References
1. Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab, Christian Wachinger  (2018). QuickNAT: Segmenting {MRI} Neuroanatomy in 20 seconds. CoRR
2. Sebastian Ruder (2017). An Overview of Multi-Task Learning in Deep Neural Networks, CVPR
3. Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, Martial Hebert (2017). Cross-stitch Networks for Multi-task Learning, CVPR
4. Alex Kendall, Yarin Gal and Roberto Cipolla, Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics, CoRR 2017
5. Sailesh Conjeti, Magdalini Paschali, Amin Katouzian, Nassir Navab (2017), Learning Robust Hash Codes for Multiple Instance Image Retrieval, MICCAI 2017