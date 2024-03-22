# HW02: Meta-heuristic Algorithm

## 簡介

使用**爬山演算法(Hill climbing, HC)** 以及**基因演算法(Genetic algorithm, GA)** 解決[01knapsack](https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html)問題，以 Python 配合 random 套件實現。

---

## (1) Hill Climbing Algorithm

以 python 實作爬山演算法，對[p06](https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html)資料集迭代 100 次後，得出收斂結果，並產生圖形。

### 簡介

#### Initilization

固定陣列大小後，在每個 index 隨機生成 0 或 1，以此筆資料進行之後動作。

#### Transition

隨機置換：隨機挑選資料 Index 將 0->1（或 1->0）。
鄰近置換：隨機挑選初始 index，再隨機選擇其左或右，將 0->1（或 1->0）。

#### Evaluation

算出 profit，將用以選擇是否留下 transition 作為結果

#### Determination

參考 evaluation 結果做出留下或淘汰的決定

### 結果

<img src="/info/HC.png" width="550"/>

---

## (2) Genetic Algorithm

以 python 實作基因演算法，對 p06 資料集迭代 100 次後，得出收斂結果，並產生圖形。

### Setting

設定必要參數：交配率(crossover rate)、突變率(mutation rate)、初始群體數量(population)、終止條件等等

### Initilization

固定陣列大小後，在每個 index 隨機生成 0 或 1

### Transition/Selection

- 挑選出 2 個染色體（兩筆表現最好的資料）
- 採用**精英挑選法** ，計算當前群體 profits 最優者，以此兩筆進行下一步

### Crossover 交配

- 交配：選定資料的其中一個位置，將資料切割，互相交換切除的基因，成為兩筆新資料
- 生成隨機機率值，若在 crossover rate 的範圍內就進行交配；若否則維持原
- **採單點交配，意指只會切割一次**

### Mutation 突變

- 突變：隨機選擇一個位址發生 0->1（或 1->0）
- 生成隨機機率值，若在 mutation rate 範圍內，就進行突變；若否則維持原狀

### Fitness 適應性

- 汰除超出負重的資料
- 留下 profit 較高的資料

### 結果

<img src="/info/GA.png" width="550"/>

## 筆記

在進行 knapsack 時需要考慮負重問題，在計算 profit 時，有許多看似優於收斂結果，實則超出負種限制的資料。

在實作 HC 時，只需在 evaluate 過程中檢測出超出負荷者，強制其 profit 為 0，就能避免收斂到超重的資料；
然而在實作 GA 時，因為「突變」的關係，還是可能會得到 profit 為 0 的資料，造成資料難以收斂。

為了解決此問題，才將 GA 適應性(fitness)納入考量。原先以為適應性就是「挑出最大 2 筆資料」的部分，然而加上 fitness 後收斂效果大幅上升。

新問題：突變仍會造成影響，有時會在迭代後期在圖片上產生低谷。

調低 mutation rate 可以防止此問題，然而如此一來似乎背離基因演算法的初衷。
