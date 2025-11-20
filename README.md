# About Dataset
取用kaggle上經過處裡的[adjusted版本](https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval-adjusted)

The dataset is the re-organized and re-labeled version of the **In-shop Clothes Retrieval Benchmark** of DeepFashion. It includes **13,752 pairs of images and masks**.

The original data was presented in the form of a deep file hierarchy and had to be re-organized as only **image** and **mask** folders under the `data` directory. All masks had three channels; they were reduced to **one channel**. Not all images had masks in the original dataset. Images without masks were discarded. You can find the script that achieves these tasks [here](#).

**Reference:**  
Liu, Ziwei, Luo, Ping, Qiu, Shi, Wang, Xiaogang, Tang, Xiaoou. *DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations.* Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

**Original source:** [DeepFashion: In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf)

You can find the notebook where this dataset is used [here](#).

**License info:** [DeepFashion License](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf)

# 資料前處理
[切分後資料](https://www.kaggle.com/code/suchiwen/cvreport/output)

# model
[pro-50](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/ER3LFtP-o-pDm25DOOBwS6IBvokjXQZIyXBSd9aB2Du8XA?e=Eu3M4W)

# faiss.index
[faiss_deepfashion_retrieval_pro-50](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/IQDDv7o_rILdRZKIidSZZ6WgAbq7fouLnlqmLzRsz9bgZDo?e=w1zQp2)
