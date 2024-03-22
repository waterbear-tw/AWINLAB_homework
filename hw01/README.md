# HW01: Dog Breed Classifier

## 簡介

引用[Kaggle](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)上 70 狗狗種類資料集，以 Python 配合 tensorflow、Keras 搭建神經網路並對資料進行訓練，訓練後再以測試集輸出預測結果，流程如下：

- 載入資料集
- 導入套件
- 資料前處理
- 以卷積神經網路訓練模組
- 預測並輸出測試結果

## 工作流程

### 資料前處理

採用 Kreas.preprocessing.image.ImageDataGenerator 類別進行，其將輸入之圖片生成一個張量 tensor，包含代表平面的二維像素點，以及代表色彩通道 RGB 的三個維度（若為黑白則只需一個維度）。

採用(224, 224, 3)作為 inputs 大小

---

### 神經網路架構(1)(2-甲)

#### 使用 ResNet50：

在搭建自己的 CNN 時發現效能總是不佳，Accuracy 時常落於 10%左右；

最終選擇 ResNet50 作為神經網路架構，得到 97%左右的 Accuray。

原先使用架構如下：

> Conv2D → MaxPool2d → Conv2D → MaxPool2d → Conv2D → Conv2D → Conv2D → MaxPool2d → Dense → Dense → Dense
> 也就是試著重現 AlexNet

> <img src="/info/alexnet.png" width="550"/>

#### Compiling setting

- Epochs: 200
- Batch size: 32
- Optimizer: Adam
- Loss function: categorical crossentropy

---

### Model Accuracy(2-乙)

<img src="/info/accuracy.png" width="550"/>
<img src="/info/loss.png" width="550"/>

### Output of Testing set (3)

詳如[output.xlsx](/output.xlsx)

<img src="/info/output.jpg" width="550"/>

---

(以下為筆記)

# 過程中的學習筆記

### 深度學習的函數類型

組合函數(combination function)、啟動函數(activation function/激勵函數)、誤差函數(error function)、目標函數(object function)

- **組合函數(Combination Funtion)**

  將上一層的輸出作為輸入，經過自身函數後會給予輸出，當函數作為神經網路中其中一層，作為向量映射功能時，就稱為組合函數。

- **啟動/激勵函數(Activation Function)**

  激勵函數的作用是，藉由引入一個非線性函數，使得神經網路預測能力提升。
  對於**反向傳播演算法(Backpropagation Algorithm)**而言，激勵函數必須可微，符合條件的函數被稱**為 Sigmoid Function**。

  > **Sigmoid v.s. Threshold**
  > 以往認為若使用 Threshold 函數，其誤差函數會是逐級函數(Stepwise Constant)，其一階導數為 0 或者不存在，則無法計算反向傳導演算法之一階梯度(Gradiant)，故以前認為 Sigmoid 較佳，連續可微之特性使得參數調整後，輸出變化幅度大，可以快速收斂；若用 Threshold 則參數變化造成之影響細微，收斂較慢。
  > 然而 1991 Sepp Hochreiter 發現 Sigmoid 具有梯度消失問題(Gradiant Vanishing)，因為 Sigmoid 之值域在(-1,1)或者(0,1)，當許多在此域內的值連續相乘後，第 n 層的梯度將趨近於 0。
  > Threshold 則因為值域範圍無此特性，故沒有梯度消失問題。

- 常用之激勵函數：

  - **Sigmoid 函數（Logistic 函數）:**

    - **適用範圍：** 主要用於二元分類問題（雙值因變數）的輸出層。
    - **特點：** 輸出範圍在 0 到 1 之間，可以將輸出解釋為概率。

  - **Softmax 函數:**

    - **適用範圍：** 通常用於多類別分類問題的輸出層（離散因變數），例如 0~9 數字識別。
    - **特點：** 將輸出轉換為概率分佈，所有類別的概率總和為 1。

  - **Tanh 函數（雙曲正切函數）:**

    - **適用範圍：** 類似於 Sigmoid，但輸出範圍在-1 到 1 之間（有限值域之連續因變數）。
    - **特點：** 在某些情況下，相對於 Sigmoid，Tanh 的表現更好。

  - **ReLU 函數（Rectified Linear Unit）:**

    - **適用範圍：** 通常用於隱藏層，可以有效處理梯度消失的問題（因變數取值為正，但沒有上限）。
    - **特點：** 輸出是輸入的正值，負值變為零。

  - **Leaky ReLU 函數:**
    - **適用範圍：** 類似於 ReLU，但對於負值的部分有一個小的斜率，有助於解決死亡神經元的問題。

  激勵函數可以理解為統計學中，廣義線性模型之連結函數(Link Funtion)。

- **誤差函數/損失函數(Error Function)**
  監督式學習中，需要一個函數去測量預測模型之輸出與實際清況之間的差異；有些非監督式學習也需要類似功能的函數。

  完美的模型誤差=0，偏離 0 之數值越小越好，意即誤差值越趨近於零，預測模型效能越好。

  交叉熵可以解釋為：映射到最可能屬於之類別的 對數，因此當實際分不等同於預測分佈時，交叉熵最小。

  - 常用的誤差函數：
    1. **均方誤差（Mean Squared Error，MSE）:**
       - 公式：$*MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2*$
       - 用於回歸問題，衡量實際值和預測值之間的平均平方差。
    2. **平均絕對誤差（Mean Absolute Error，MAE）:**
       - 公式：$*MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|*$
       - 也用於回歸問題，衡量實際值和預測值之間的平均絕對差。
    3. **二元交叉熵（Binary Cross-Entropy）:**
       - 公式：$*BCE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]*$
       - 用於二元分類問題，其中$*y_i*$和$*\hat{y}_i*$分別表示實際值和預測值。
    4. **分類交叉熵（Categorical Cross-Entropy）:**
       - 公式：$*CCE = -\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij} \log(\hat{y}_{ij})*$
       - 用於多類別分類問題，其中$*{y}_{ij}*$和$*\hat{y}_{ij}*$分別表示實際和預測的類別概率。
    5. **Hinge Loss:**
       - 主要用於支持向量機（Support Vector Machines，SVM）和一些二元分類問題。
       - 公式：$HingeLoss = \max(0, 1 - y \cdot \hat{y})$
       - $y$ 和 $*{\hat y}*$分別表示實際類別和預測的分數。
    6. **Huber Loss:**
       - 具有均方誤差和平均絕對誤差的優點，用於回歸問題。
       - 公式：$HuberLoss = \begin{cases} \frac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta), & \text{otherwise} \end{cases}$
       - $\delta$ 是一個超參數，用於控制平方誤差和絕對誤差的過渡。

- **目標函數(Object Function)**
  最佳化理論中，我們欲對其進行最大化或者最小化的那個函數；當我們在 train 一個 model，就是希望讓 Object Function 有最佳的效能表現。

  模型普適化差：

  Model 在訓練集可以達到一定的準確度，但在真實使用時不行的狀況稱之。此時會採用正規化規範模型，減少過度擬合的情況。這種狀況的 Object Function 會是 Loss Function 以及正規函數的加成。

### 其他概念

**批量 Batch**

有兩種概念，且二者有緊密關聯：

1. 對於訓練模型，Batch 指的是將資料都處理完後，一次性更新權重或參數的估計值，這個一次性更新的流程稱之。
2. 對於在模型被用以訓練的資料而言：Batch 是指一次輸入模型中計算的資料量稱之。

**線上學習/離線學習 Online learning and Offline Learning**

資料在訓練後可以反覆取得者，為離線學習；每個觀測值在使用後被丟棄則是線上學習。

Offline Learnring 之優點：

1. 對於任何固定個數的參數，可以直接計算出目標函數，因此很容易驗證模型訓練是否朝目標發展
2. 計算精度可以達到任何合理之程度
3. 可以使用各種不同的演算法，避免局部最佳化之問題
4. 可以採用訓練、驗證、測試三分法，針對模型的普適化進行驗證
5. 可以計算預測值及信賴區間

**偏移/ 門檻值 Bias**

在需要加入激勵函數的模型中，通常會對隱藏層或是輸出層加入偏移值；偏移項通常就是回歸之中的截距項。

> ……如果沒有偏移項，表示超平面的位置會被限制住而必須要通過原點；若多個神經元都需要各自的超平面，就會嚴格限制模型的靈活性。

**標準化資料**

標準化包含下列三個動作：

1. 重調整 Rescaling

   逕以乘法或加上常數來調整函式，例如攝氏華氏變換

2. 正規化 Normalization

   將向量變成單位向量，即將向量除以其範數 norm。深度學習中，通常以全距為範數，將向量減去最小值後，除以全距，得到座落於(0,1)的數值，此過程即為正規化。

3. 標準化 Standardization

   將向量除以其位置和規模的度量。（？？？）例如，一個希望他遵守常態分佈的向量，可以減去均值，並除以其變異量數來標準化資料，進而達成目的。

標準化該如何運用？

- 激勵函數值域(0,1)時，正規化資料到[0,1]較為合理。
- 正規化能使得資料更穩定，意即資料的值域範圍可以因為正規化而縮小。

#### 梯度下降法

（related to 偏微分）
[關於更深度的說明](https://t.ly/V2QOk) **by** [chih-sheng-huang821](https://chih-sheng-huang821.medium.com/)

誤差函數中，誤差最小的點就是最佳解之所在。

所以在最佳化的過程，必須用一些方法來「走到最低點」；至於是否是最好的方法，就看當前方法是否「真的走到了最低點」以及「下降的速度夠不夠快」來決定。一般做到這件事的方法即是**「梯度下降法( Gradient Descent Method )」**

#### 反向傳播演算法 (Backpropagation)

（related to 偏微分）
當神經網路只有一層時，可以使用梯度下降演算法；當神經網路有許多層時，則使用反向傳播演算法使損失函數收斂。
