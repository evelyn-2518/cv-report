# DeepFashion In-Shop Clothes Retrieval System
> 電腦視覺課程期末專題

這是一個基於 **PyTorch** 與 **FAISS** 建構的服飾以圖搜圖系統。我們實作了 Baseline (ResNet50) 與 Advanced (Metric Learning) 兩種模型，並透過 Gradio 建立互動式介面。

## About Dataset
取用kaggle上經過處裡的[adjusted版本](https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval-adjusted)

The dataset is the re-organized and re-labeled version of the **In-shop Clothes Retrieval Benchmark** of DeepFashion. It includes **13,752 pairs of images and masks**.

The original data was presented in the form of a deep file hierarchy and had to be re-organized as only **image** and **mask** folders under the `data` directory. All masks had three channels; they were reduced to **one channel**. Not all images had masks in the original dataset. Images without masks were discarded. You can find the script that achieves these tasks [here](#).

**Reference:**  
Liu, Ziwei, Luo, Ping, Qiu, Shi, Wang, Xiaogang, Tang, Xiaoou. *DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations.* Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

**Original source:** [DeepFashion: In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf)

You can find the notebook where this dataset is used [here](#).

**License info:** [DeepFashion License](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf)

## 專案資源與檔案下載 (Project Resources)

為了方便重現實驗結果，我們提供了處理好的資料集、訓練權重與預先建立的索引檔案。點擊下方連結即可存取：

| 類別 (Category) | 內容說明 (Description) | 檔案連結 (Download) |
| :--- | :--- | :--- |
| **資料前處理** | 已經過切分與清洗的訓練/驗證資料集 | [切分後資料](https://www.kaggle.com/code/suchiwen/cvreport/output) |
| **Model Weights** | **Baseline (pro-50)**: ResNet50 分類模型權重 | [pro-50](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/ER3LFtP-o-pDm25DOOBwS6IBvokjXQZIyXBSd9aB2Du8XA?e=Eu3M4W) |
| | **Advanced**: ArcFace + Triplet Loss 精進模型權重 | [advanced](https://drive.google.com/file/d/1Krbx2FEd3dFD8tpKgR2dBX-PlXkgcTok/view?usp=sharing) |
| **FAISS Index** | 用於快速檢索的向量索引檔 | [pro-50 Index](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/IQDDv7o_rILdRZKIidSZZ6WgAbq7fouLnlqmLzRsz9bgZDo?e=w1zQp2)<br>[advanced Index](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/IQDwztyLocY3SYY8ONYO3kBYAS7aadl_9FGyUeRNikdkNBc?e=fucdIf) |
| **Feature Vectors** | 預先提取的特徵向量文件 (Numpy/Pickle) | [advanced Vectors](https://changgunguniversity-my.sharepoint.com/:u:/g/personal/b1228022_cgu_edu_tw/IQC7v5QjOzpWTophOeke5PKsAdbPQ4k947YE1zhHfZqa0Tw?e=df4mKO) |

> **注意**：請將下載的模型權重放入專案根目錄，或修改 `config` 中的路徑以符合您的設定。


## 系統架構 (Architecture)

本系統包含兩個主要的模型路徑，用於驗證 Metric Learning 在檢索任務上的有效性：

### 1. Baseline Model (pro-50)
* **架構**：ResNet50 (Pre-trained on ImageNet)
* **方法**：標準分類任務 (Classification Task)
* **特徵提取**：提取 2048 維特徵向量。

### 2. Advanced Model
* **架構**：ResNet50 Backbone + Embedding Layer (512-dim)
* **方法**：度量學習 (Metric Learning)
* **損失函數**：ArcFace Loss + Triplet Loss 混合優化。

## 快速實作

### 1. 環境安裝
```bash
pip install torch torchvision numpy pandas pillow faiss-cpu gradio
```
### 2. 啟動 Web UI Demo
確保您已下載上述 Model 與 FAISS Index 檔案，接著執行：

[UI程式碼](CVUI.ipynb)

## 實驗結果
我們使用 Recall@K 與 t-SNE 進行評估。雖然 Baseline 在全量數據評估下有 Overfitting 現象，但 Advanced 模型在 t-SNE 可視化中展現了更緊密的群聚效果。
