# DA-GET-BFM Proposal v4

**Dynamics Aware Cross Embodiment Behavior Foundation Model with Zero-Shot Capabilities**

版本 v4 · 2026-11-19 · branch `feature/proposal-v4`

根據的標記方式:**[R#]** 是本 repo 的程式碼(file:line)或實測數字;**[n]** 是文末第 n 篇文獻(查核過原文或官方頁面,查不到的地方在 reference 註明);**[H]** 是沒有直接先例的假設,附驗證實驗,結果出來前不當成結論。

---

## 0. 目標與成功標準

目標就是 README 的標題:**Dynamics Aware Cross Embodiment Behavior Foundation Model with Zero-Shot Capabilities**。其中 zero-shot 有兩層:

- **Task zero-shot**:不重新訓練,用 goal、tracking、reward 下 prompt(BFM-Zero [2],建立在 FB-CPR [3] 與 forward-backward representations [4] 上)。
- **Embodiment zero-shot**:用在沒訓練過的機器人上(GET-Zero [1] 的目標)。

**最終成功標準是 unseen 機器人上的 BFM 指標(5.4)**。Encoder 的物理指標只是診斷工具,用來解釋 BFM 成功或失敗的原因。

## 1. 更正前幾版的錯誤

查核後發現 v2/v3 有以下錯誤:

1. **GET-Zero 的 20%**:那是完整方法(graph encoding + self-modeling)對上最佳 baseline 的提升。Self-modeling loss 單獨的效果是:有 graph encoding 時約 +5.1%,沒有時 −4.6% [1, Table I]。
2. **Embodiment scaling**:Ai et al. 沒有擬合 power law。他們的結果是:訓練機器人從 5% 增加到 100%,held-out reward 約翻倍;四足和六足約在 100 台時飽和,人形還在上升 [38]。
3. **MetaMorph**:沒有做「訓練形態數量或多樣性」的 ablation [29]。
4. **SWAT**:zero-shot 並沒有勝過 Amorpheus [33]。
5. **v3 的超參數是我自己訂的**(α=0.5、LoRA rank 8、15% mask、0.5–100 kg、k~U(1.5,4)、learning rate 數值、每輪 64 個形態)。本版全部改成引用值、repo 既有設定或實測值,否則標成 [H]。

## 2. 現況證據(repo)

| ID | 發現 | 位置或量測 |
|---|---|---|
| R1 | FD、R、E 的 target 除以「每個 record 的資料 RMS」。同一台機器人的不同 source 是不同 record;physics replay 會複製 record,並在新 episode 進來時重算 scale | `model/training/episodes.py:1077`、`dataset.py:513,548,575`、`rl/physics_replay.py:131,265` |
| R2 | Control period 隨 source 不同:synthetic、MuJoCo 0.01 s,Hub 資料 1/fps(例如 30 Hz),RL 0.02 s;token 裡沒有 dt | `sources.py:226,843,950`、`rl/envs.py:96`、`episodes.py:1021` |
| R3 | Rigid-body features 只描述每個 joint 的 child body;base 的質量、CoM、慣量不在任何輸入裡 | `model/physics.py:257` |
| R4 | 接觸狀態既不是輸入也不是預測目標,但 R labels 依賴它(忽略接觸時誤差 > 100%) | `docs/dynamics.md` |
| R5 | Heads 比 encoder 大(實測):small 是 encoder 1.12M、heads 1.45M;medium 是 6.55M 對 5.75M。FD、ID、R、R⁻¹、E 各有一層私有 graph-attention;BFM 不使用這些 heads;RL 時 heads 的 lr 1e-4 高於 encoder 的 3e-5 | `model/pretraining.py:255,338–350`、`docs/bfm.md:427–428` |
| R6 | B/D 的 state path 只用 M 的對角線,normalize 後 mass bias 失效 | `model/modeling_bfm.py:645` |
| R7 | 每個 probe 或 proposal variant 都是獨立 sample,同一個輸入要重複 encode | `dataset.py:321` |
| R8 | 訓練形態只有 Go1、ANYmal D、G1、H1、四旋翼與 G1 手臂,加上 mass ±20%、length ±10% 擾動 | `configs/data/*.json`、`DEFAULT_RANDOMIZATION` |
| R9 | 四台機器人實測:質量 13.1–59.3 kg、L 0.31–1.00 m;peak / 靜態站立 torque 的中位數為 ANYmal D 3.4、H1 7.8、Go1 12.6、G1 13.3,最小值 2.3 | 用 `ground_contact.json` 的站姿,接觸力取最小範數分配 |
| R10 | BFM 的 z 只在同一台機器人內配對;action = `nominal + 1.25·a`,K_p = 2τ_peak;README 寫明「沒有東西對齊跨機器人的行為」 | `rl/agent.py:218`、`ground_sim.py:457`、`docs/bfm.md` |

R9 只有四台機器人,而且接觸力分配不唯一,只能當量級參考。

## 3. 提案

### A. Target 與輸入只取決於物理

**A1. 用 physics-derived normalization 取代 data RMS**

- 根據:R1——同一個輸入在不同 source 有不同 target,MSE 的最佳預測是條件期望,這部分誤差無法消除;遇到沒見過的機器人,scale 也無法預測。Gauss 原理:constrained acceleration 是 (q̈ − a)ᵀM(q̈ − a) 的最小解 [6, Eq. 6],M-metric 是這類動力學的自然度量。
- 做法:兩個候選,用 5.6 的 ablation 決定 [H]:(a) 只用輸入就算得出的 scale(τ_peak/M_ii、g);(b) M-norm loss。

**A2. 統一成 50 Hz**

- 根據:R2;BFM-Zero 用 50 Hz 的 PD control,motion 也重採樣到 50 Hz [2] (repo 的 `docs/bfm.md`)。

**A3. 無因次化(dynamic similarity)**

- 根據:R9(質量差 4.5 倍、L 差 3 倍);Froude 數相等時哺乳類以 dynamically similar 的方式運動,論文自己限定在 cursorial 類別最成立 [14];Pi theorem [15] (內文未查核);GenLoco 讓 PD gain 與總質量線性縮放 [36]。
- 做法:以 m_total、L、√(L/g) 為單位,絕對尺度放進 root token(C1)。對 encoder 是否有幫助是 [H]。

### B. 物理要學在 encoder 裡,而不是 heads 裡

**B0. 先量測,再決定**

- 根據:R5——heads 的容量足夠自己做物理計算。但文獻兩個方向都有證據:
  - 支持強 head:SimCLR 中非線性 projection head 之前的表示,linear eval 高出 >10% [46];MAE 中 decoder 夠深時 encoder 的 linear probe 可高約 8%,到 8 層後持平 [47]。
  - 支持結構化 readout:DeLaN 的物理結構對沒見過的速度外推較好 [16];E2C 把 latent dynamics 限制成 A(z)z + B(z)u + o(z) 以便控制 [17]。
- 量測:逐層 linear probe [42];用 control task 算 selectivity [43];random-encoder baseline [44, 45] (這兩篇用 MLP probe,所以 linear 與 MLP probe 都跑)。
- 決策規則:如果「現行 heads + 凍結的隨機 encoder」的 loss 接近完整訓練,代表物理學在 heads 裡,才需要做 B1。

**B1. Physics interface:結構化的線性 readout**

- 根據:
  - Ĝ 的結構:joint block G_jj = S[M⁻¹ − M⁻¹Jᵀ(JM⁻¹Jᵀ)⁻¹JM⁻¹]Sᵀ(`docs/dynamics.md`)。中括號等於 M^{-½}(I − Π)M^{-½},Π 是正交投影,所以 G_jj 對稱半正定。參數化成 Ĝ = UUᵀ + diag(softplus) 就能保證這點,作法同 DeLaN 以 LLᵀ 保證正定 [16]。
  - FD 寫成 q̈ = f̂ + Ĝτ:這是 manipulator equation 本身的 control-affine 形式 [5],也是 E2C 的 A、B、o 結構 [17]。物理結構帶來守恆或較好的長期預測 [18, 19]。
  - ID 與 R⁻¹:閉式解用到 Moore–Penrose 反矩陣 [6, Eq. 5a];不獨立約束的 projection operator [8];以 QR 消去 constraint force [7]。
  - 監督訊號沒有用足:每個 frame 都存了完整的 G,但目前每個 sample 只用一個隨機 δτ 監督(`episodes.py:842`)。
- [H]:E 改用「預測模型的可微一步 rollout + 解析 FK」沒有直接先例;純線性 readout 的準確度代價用「純線性」對「線性 + 小殘差」的 ablation 量測。

**B2. BFM 直接使用這個介面**

- 根據:加性 attention bias 本來就是注入結構資訊的機制(Graphormer [48]、GET-Zero [1]、repo 現有的 mass bias);與 policy 共用表示的輔助任務能改善 RL [27]。
- [H]:把預測的 Ĝ 當成 actor 的 attention bias,用有/無介面 bias 的 ablation 驗證。

**B3. 調整 RL 階段 heads 的學習率**

- 根據:R5。[H]:`lr_physics_heads` ≤ `lr_encoder`,用 B0 的 probe 確認 RL 之後 H 是否仍保有物理資訊。

### C. 讓 encoder 看得到關鍵物理量

**C1. Root token**:R3;MetaMorph 把 root(torso)也當 token,每個 module 的輸入有形狀、密度、joint 參數與 gear [29];URMA 的 description 包含整台機器人的質量與尺寸 [37]。

**C2. 接觸機率 head**:R4;Ji et al. 讓 estimator 與 policy 一起訓練,輸出 base 線速度、腳高與接觸機率,接觸機率對最終性能影響最小但讓學習更穩、更快 [20];Lin et al. 從本體感測學出接觸事件 [21]。Labels 來自模擬器或感測器;推斷出的接觸只保留通過 repo confidence 篩選的 frame(`docs/dynamics.md`:82–97% 的接觸組合正確)。

**C3. Centroidal targets**:centroidal momentum matrix 把廣義速度映到 centroidal momentum [9, 10];ḣ 等於接觸 wrench 加重力的合力,同一模型用在人形 Atlas 與四足 LittleDog [11];single rigid body 模型用在四足 MPC [12]。[H]:把它當 encoder 的輔助目標沒有先例,用 ablation 驗證。

### D. 形態覆蓋(embodiment zero-shot)

**D1. 程序化形態生成器**

- 根據:R8;Ai et al.:1,012 台程序化生成的機器人,訓練比例 5% → 100% 時 held-out reward 約翻倍,真機 Go2、H1 可 zero-shot [38];GenLoco:隨機化四足形態後可 zero-shot 到 10 個沒見過的模擬機器人(含膝蓋結構不同的 ANYmal)與 3 台真機 [36];MetaMorph:100 台訓練、可 zero-shot 到另 100 台但性能下降 [29];ManyQuadrupeds:一個 policy 控制 2–200 kg 的 16 種四足 [39]。
- 反例:URMA 對 H1 與 Talos 人形完全沒有初始遷移 [37];Ai et al. 的人形結果還沒飽和 [38]。所以人形風險最高、需要最多形態。
- 生成參數直接沿用文獻:Ai et al. Table 6 [38] (膝關節數、link scale、thigh/calf、腳尺寸、torso scale、膝關節限位 scale);GenLoco Table 1 [36] (整體尺寸、base 密度、thigh/calf、link mass);慣量由幾何 × 密度算出,並檢查 pseudo-inertia 正定(三角不等式)[13]。
- 致動器:找不到已發表的 torque limit 生成規則(Ai et al. 每類固定 [38];UNIMAL 的 gear 與質量無關 [30])。PD gain 依 GenLoco 隨質量縮放 [36];torque limit 從 R9 的實測範圍(peak/站立 torque 介於 2.3 與約 13)抽樣——只有四台機器人的根據,標 [H],並在 L4 機器人上檢查。

**D2. 抽樣平衡**:R7;temperature sampling q_i ∝ p_i^α [51],XLM-R 最佳 α=0.3 [52],Arivazhagan et al. 用 T=5(等同 α=0.2)[53];Octo 依資料集大小設權重再手動調整 [54]。做法:α 從 0.2–0.3 開始在 val 上調;每個 state 只 encode 一次,所有 variants 共用;依 family 平衡是 [H] (動機來自 [38] 的人形結果)。

**D3. 左右對稱 augmentation(可選)**:augmentation 讓收斂更快、行為更好 [40];morphological symmetry 的理論 [41]。

**D4. 拓撲 bias:證據互相衝突,交給實驗**:忽略拓撲的 Amorpheus 在 multi-task 勝過 SMP,但 zero-shot 結果初步且變異大 [31, 32];SWAT 的結構 embedding 對 multi-task 與 fine-tune 有幫助、對 zero-shot 沒有 [33];MetaMorph 反轉 DFS 順序後 zero-shot 下降約 75% [29];ModuMorph 以只由形態決定的 attention 加 hypernetwork,zero-shot 勝過 MetaMorph [34];GET-Zero 的 graph encoding 有幫助 [1]。做法:在 L2/L3 比較 (1) 現行 GET bias、(2) 移除拓撲 bias、(3) ModuMorph 式 modulation。

### E. 對錯誤或缺漏的規格保持穩健

**E1. History 0.5 s**:RMA 用 50 步(0.5 s)history 推估質量、motor strength、摩擦等 extrinsics [22];Lee et al. 的 TCN student 從 history 推斷 privileged 資訊 [23];GenLoco 不給形態參數,只用 15 步 history(30 Hz)[36];repo 目前 H=3,在 50 Hz 下只有 60 ms。做法:50 Hz 下取 25 步,加上 privileged deviation head。

**E2. Spec randomization**:dynamics randomization 讓 policy 不用真機訓練也能遷移 [25];domain randomization [26];actuator 難以解析建模 [24],repo 的預設值也是猜的(armature 0.01、τ_cont = 0.5×peak)。[H]:隨機化的是「給網路的規格描述」而不是模擬器參數,沒有先例,用 5.5 驗證。

**E3. Masked physics modeling**:MaskDP mask state 與 action token,可泛化到多個下游任務並 zero-shot 遷移到新的 BC 任務 [57];MTM 的表示能加速 RL [56];mask 比例的調整範圍參考 BERT 15% [55] 與 MAE 75% [47];AnyMorph 不靠手寫描述、從資料學出形態表示,但前提是 token ID 在機器人間一致 [35]。[H]:對形態欄位做 masking 沒有先例。

### F. Unseen 機器人的 test-time adaptation

**F1. 用物理 loss 自監督調整**:PAD 讓 encoder 與 inverse-dynamics head 共用,部署時不用 reward、只用自監督 loss 更新,36 個環境中 31 個泛化變好 [59];TTT 用自監督任務在測試時更新模型 [58]。Repo 的優勢:任何 URDF 都能用 synthetic 或 MuJoCo source 產生 exact labels,不需要 torque。防止表示漂移:L2-SP 把 source task 的下降從 −24.1 降到 −5.3 [61];LwF [62];LoRA rank 在 1–8 掃描(論文 Table 6 的範圍,作者也說明小 rank 不一定適用所有任務)[60]。[H]:PAD 處理的 distribution shift 主要是視覺的,用在 embodiment shift 上未經證實。

### G. BFM 層的 zero-shot

**G1. Action 參數化**:查核過的跨形態工作都用 residual position——URMA 與 Ai et al. 是 q_nominal + σ·a,σ 依機器人或類別而定 [37, 38];GET-Zero 在 PD target 上加 delta [1];GenLoco 的 gain 隨質量縮放 [36];R10:repo 對所有機器人都用 1.25 rad。做法:保留 residual 形式,比較「σ 依類別」與「gain 隨質量」;前幾版的「依 joint range 縮放」沒有先例,刪除。

**G2. RL 也在程序化形態上做**:GenLoco 直接在隨機化形態上做 RL [36];Ai et al. 先為每台機器人訓練 PPO expert 再蒸餾到 URMA [38]。[H]:在數百種形態上做 FB-CPR 沒有先例。

**G3. 跨形態 prompt**:R10;以對應 keypoints 加 IK 做 retargeting [28]。做法:先把 prompt retarget 到新機器人,再用 BFM-Zero 原本的 tracking 或 goal inference [2],推論方式不變。[H]:另外學一個 task-space 的 B。

## 4. Training

| 階段 | 設定 | 根據 |
|---|---|---|
| 資料 | D1 生成器;split manifest 固定成檔案;全部 50 Hz;labels 含 G、P、centroidal、contact 與其 confidence | A2、D1 |
| Stage 1 loss | FK、R/G̃、FD/ID(只用 exact data)、centroidal、contact、E、mask、sysID | B1、C、E |
| Loss 權重 | Baseline 用 repo 預設(全為 1),再比較 uncertainty weighting [49] 與 GradNorm [50] | |
| Optimizer | AdamW [63];lr、warmup、batch 從 repo 文件的設定出發(`docs/training.md`:5e-4、100、64),在 val 上調整 | |
| 抽樣 | α ∈ {0.2, 0.3} [52, 53];每個 state 只 encode 一次 | D2 |
| History | 0.5 s | E1 [22, 36] |
| 選模型 | 只看 val split;L3/L4 test 最後只用一次 | 獨立 held-out set,同 [29, 38] |
| Stage 2 RL | FB-CPR,超參數用 BFM-Zero 的值(`docs/bfm.md` [2]);physics replay;B3;只用訓練形態 | G2 |
| Stage 3 TTA | URDF → 合成資料 → LoRA [60] + L2-SP [61] 或 LwF [62];用新機器人的合成 val 決定何時停止 | F1 |

## 5. Evaluation

### 5.1 Splits

| 層級 | 內容 | 根據 |
|---|---|---|
| L0 | 訓練形態的 held-out episodes | |
| L1 | 參數變化:MetaMorph 的六類(armature、density、damping、gear、module shape、joint angle)加尺寸外推 | [29]、A3 |
| L2 | 同 family 的新拓撲:膝關節數、腳尺寸等 | [38] |
| L3 | 整個 family 留出,例如用四足與人形訓練、測六足 | |
| L4 | 真實機器人模型:Go2、H1([38] 的真機)、Talos(URMA 失敗的機器人 [37])、LocoMuJoCo 的其他人形 | |

L1–L3 各分 val 與 test,held-out 比例照 [38] 用 20%。

### 5.2 Encoder 物理指標(全部無因次)

Operator 與 response 的 relative error;只在 exact data 上算的 FD/ID 誤差;centroidal 誤差;contact F1;FK 誤差 / L。每一層級都報告 gap Lk − L0。

### 5.3 物理是否在 encoder 裡(B0)

1. 逐層 linear probe [42],並計算 selectivity [43]。
2. Random-encoder baseline [44, 45],linear 與 MLP probe 都跑。
3. Head re-init transfer:凍結 encoder,重新初始化 heads 只在 L0 上訓練,再到 L2/L3 測。
4. Counterfactual:改變一個 physics token,與 `constrained_response` 解析算出的 ΔG 比較 [5]。

### 5.4 BFM 指標(兩種 zero-shot 都要量)

- Task zero-shot:repo 已有的 `tracking_mpjpe`、`tracking_emd`、`goal_mpjpe`、`getting_up_rate`(`docs/bfm.md`),再加上 reward prompts [2]。
- Embodiment zero-shot:在 L1–L4 量同一套指標,並與該機器人的 specialist policy 比較(GenLoco 就是這樣與 robot-specific policy 比較 [36])。
- 分別報告 zero-shot 與 TTA 之後的結果。

### 5.5 穩健性

規格錯誤時的性能下降:armature 用預設值、τ_cont 用預設值、mass ±20%;GenLoco 的 mass 隨機化範圍 ×0.5–1.5 [36] 作為較大範圍的參照。

### 5.6 Ablation(每一列對應一個提案)

| 項目 | 比較 | 對應 |
|---|---|---|
| Heads | 現行 mixer、線性介面、線性 + 殘差 | B0、B1 |
| Normalization | data RMS、physics-derived、M-norm | A1 |
| 輸入尺度 | 絕對、無因次 | A3 |
| 物理量 | root token、contact、centroidal 各自開/關 | C |
| 拓撲 bias | 三種擇一 | D4 |
| 形態數 | 5% → 100%,作法同 [38] | D1 |
| 規格 | spec randomization、masking | E2、E3 |
| BFM | actor 有/無介面 bias | B2 |

### 5.7 統計

多個 seed:Henderson et al. 只換 seed、各 5 個 seed 的兩組就得到 p = 0.0016 的「顯著」差異 [64]。信賴區間用 stratified bootstrap CI 與 IQM [65],並以「機器人」為單位 bootstrap。Scaling 只報告趨勢、不預設 power law(power law 是語言模型的結果 [66],[38] 也沒有擬合)。

## 6. Roadmap 與決策點

| 階段 | 內容 | 決策點 |
|---|---|---|
| 0 | 建立 splits 與 5.3 的診斷,先在現有模型上量;建立 L0–L4 的 BFM baseline | 物理確實學在 heads 裡才做 B1,否則以 A、C、D 為主 |
| 1 | A1–A3、修正 R6、D2 | L1 的 gap 是否縮小 |
| 2 | D1,並掃描形態數 | Held-out 是否隨形態數改善;人形是否飽和 |
| 3 | 依階段 0 結果做 B1/B2;C1–C3;D4 | L2/L3 與 BFM 指標 |
| 4 | E1–E3、F1 | 5.5 的穩健性與 L4 |
| 5 | G1–G3 | Unseen 機器人上的 task zero-shot |

## 7. 待驗證的假設 [H]

| 編號 | 假設 |
|---|---|
| A1 | physics-derived scale 與 M-norm 哪個較好 |
| A3 | 無因次化對 encoder 有幫助 |
| B1 | E 改用可微 rollout;純線性 readout 的準確度代價 |
| B2 | 用 Ĝ 作 actor 的 attention bias |
| B3 | heads 的 lr ≤ encoder 的 lr |
| C3 | centroidal 作為輔助目標 |
| D1 | 以 R9 的實測範圍決定 torque limit |
| D2 | 依 family 平衡抽樣 |
| E2 | 隨機化規格描述 |
| E3 | 對形態欄位做 masking |
| F1 | PAD 式自監督調整用在 embodiment shift |
| G2 | 在大量形態上做 FB-CPR |
| G3 | 學 task-space 的 B |

## 8. References

† 表示只核對到摘要或書目、內文未讀到。[2]–[4] 引自 README,這次沒有重新查核。

**專案基礎**

1. A. Patel, S. Song. GET-Zero: Graph Embodiment Transformer for Zero-Shot Embodiment Generalization. ICRA 2025, pp. 14262–14269. DOI 10.1109/ICRA55743.2025.11127922; arXiv:2407.15002.
2. Y. Li et al. BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning. arXiv:2511.04131, 2025.
3. A. Tirinzoni et al. Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models. ICLR 2025. arXiv:2504.11054.
4. A. Touati, Y. Ollivier. Learning One Representation to Optimize All Rewards. NeurIPS 2021.

**剛體與接觸動力學**

5. R. Featherstone. Rigid Body Dynamics Algorithms. Springer, 2008. DOI 10.1007/978-1-4899-7560-7.
6. F. E. Udwadia, R. E. Kalaba. A new perspective on constrained motion. Proc. R. Soc. Lond. A 439(1906):407–410, 1992. DOI 10.1098/rspa.1992.0158.
7. M. Mistry, J. Buchli, S. Schaal. Inverse dynamics control of floating base systems using orthogonal decomposition. ICRA 2010, pp. 3406–3412. DOI 10.1109/ROBOT.2010.5509646.
8. F. Aghili. A unified approach for inverse and direct dynamics of constrained multibody systems based on linear projection operator. IEEE T-RO 21(5):834–849, 2005. DOI 10.1109/TRO.2005.851380. †
9. D. E. Orin, A. Goswami, S.-H. Lee. Centroidal dynamics of a humanoid robot. Autonomous Robots 35(2–3):161–176, 2013. DOI 10.1007/s10514-013-9341-4. †
10. D. E. Orin, A. Goswami. Centroidal Momentum Matrix of a humanoid robot: Structure and properties. IROS 2008, pp. 653–659. DOI 10.1109/IROS.2008.4650772. †
11. H. Dai, A. Valenzuela, R. Tedrake. Whole-body motion planning with centroidal dynamics and full kinematics. Humanoids 2014, pp. 295–302. DOI 10.1109/HUMANOIDS.2014.7041375.
12. J. Di Carlo, P. M. Wensing, B. Katz, G. Bledt, S. Kim. Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control. IROS 2018. DOI 10.1109/IROS.2018.8594448.
13. P. M. Wensing, S. Kim, J.-J. E. Slotine. Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification: A Statistical Perspective on the Mass Distribution. IEEE RA-L 3(1):60–67, 2018. DOI 10.1109/LRA.2017.2729659.
14. R. McN. Alexander, A. S. Jayes. A dynamic similarity hypothesis for the gaits of quadrupedal mammals. J. Zool. 201(1):135–152, 1983. DOI 10.1111/j.1469-7998.1983.tb04266.x.
15. E. Buckingham. On Physically Similar Systems; Illustrations of the Use of Dimensional Equations. Phys. Rev. 4(4):345–376, 1914. DOI 10.1103/PhysRev.4.345. †

**結合物理結構的學習**

16. M. Lutter, C. Ritter, J. Peters. Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning. ICLR 2019. arXiv:1907.04490.
17. M. Watter, J. T. Springenberg, J. Boedecker, M. Riedmiller. Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images. NIPS 2015. arXiv:1506.07365.
18. S. Greydanus, M. Dzamba, J. Yosinski. Hamiltonian Neural Networks. NeurIPS 2019. arXiv:1906.01563.
19. M. Cranmer et al. Lagrangian Neural Networks. ICLR 2020 DeepDiffEq Workshop. arXiv:2003.04630.

**足式機器人學習**

20. G. Ji, J. Mun, H. Kim, J. Hwangbo. Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE RA-L 7(2):4630–4637, 2022. DOI 10.1109/LRA.2022.3151396.
21. T.-Y. Lin, R. Zhang, J. Yu, M. Ghaffari. Legged Robot State Estimation using Invariant Kalman Filtering and Learned Contact Events. CoRL 2021, PMLR 164:1057–1066.
22. A. Kumar, Z. Fu, D. Pathak, J. Malik. RMA: Rapid Motor Adaptation for Legged Robots. RSS 2021. DOI 10.15607/RSS.2021.XVII.011.
23. J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, M. Hutter. Learning quadrupedal locomotion over challenging terrain. Science Robotics 5(47):eabc5986, 2020. DOI 10.1126/scirobotics.abc5986.
24. J. Hwangbo et al. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26):eaau5872, 2019. DOI 10.1126/scirobotics.aau5872.
25. X. B. Peng, M. Andrychowicz, W. Zaremba, P. Abbeel. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA 2018, pp. 3803–3810. DOI 10.1109/ICRA.2018.8460528.
26. J. Tobin et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017, pp. 23–30. DOI 10.1109/IROS.2017.8202133.
27. M. Jaderberg et al. Reinforcement Learning with Unsupervised Auxiliary Tasks. ICLR 2017. arXiv:1611.05397.
28. X. B. Peng, E. Coumans, T. Zhang, T.-W. Lee, J. Tan, S. Levine. Learning Agile Robotic Locomotion Skills by Imitating Animals. RSS 2020. DOI 10.15607/RSS.2020.XVI.064.

**跨形態控制**

29. A. Gupta, L. Fan, S. Ganguli, L. Fei-Fei. MetaMorph: Learning Universal Controllers with Transformers. ICLR 2022. arXiv:2203.11931.
30. A. Gupta, S. Savarese, S. Ganguli, L. Fei-Fei. Embodied intelligence via learning and evolution. Nat. Commun. 12:5721, 2021. DOI 10.1038/s41467-021-25874-z.
31. W. Huang, I. Mordatch, D. Pathak. One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control. ICML 2020, PMLR 119:4455–4464.
32. V. Kurin, M. Igl, T. Rocktäschel, W. Böhmer, S. Whiteson. My Body is a Cage: the Role of Morphology in Graph-Based Incompatible Control. ICLR 2021. arXiv:2010.01856.
33. S. Hong, D. Yoon, K.-E. Kim. Structure-Aware Transformer Policy for Inhomogeneous Multi-Task Reinforcement Learning. ICLR 2022. OpenReview fy_XRVHqly.
34. Z. Xiong, J. Beck, S. Whiteson. Universal Morphology Control via Contextual Modulation. ICML 2023, PMLR 202:38286–38300.
35. B. Trabucco, M. Phielipp, G. Berseth. AnyMorph: Learning Transferable Polices By Inferring Agent Morphology. ICML 2022, PMLR 162:21677–21691.
36. G. Feng et al. GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots. CoRL 2022, PMLR 205:1893–1903.
37. N. Bohlinger et al. One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion. CoRL 2024, PMLR 270:3356–3378.
38. B. Ai et al. Towards Embodiment Scaling Laws in Robot Locomotion. CoRL 2025, PMLR 305:3483–3515. arXiv:2505.05753.
39. M. Shafiee, G. Bellegarda, A. Ijspeert. ManyQuadrupeds: Learning a Single Locomotion Policy for Diverse Quadruped Robots. ICRA 2024, pp. 3471–3477. DOI 10.1109/ICRA57147.2024.10610155.
40. M. Mittal, N. Rudin, V. Klemm, A. Allshire, M. Hutter. Symmetry Considerations for Learning Task Symmetric Robot Policies. ICRA 2024, pp. 7433–7439. DOI 10.1109/ICRA57147.2024.10611493.
41. D. Ordoñez-Apraez et al. Morphological Symmetries in Robotics. IJRR 44(10–11):1743–1766, 2025. DOI 10.1177/02783649241282422.

**表示與 probing**

42. G. Alain, Y. Bengio. Understanding intermediate layers using linear classifier probes. ICLR 2017 Workshop. arXiv:1610.01644.
43. J. Hewitt, P. Liang. Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP 2019, pp. 2733–2743. DOI 10.18653/v1/D19-1275.
44. K. Zhang, S. R. Bowman. Language Modeling Teaches You More than Translation Does: Lessons Learned Through Auxiliary Syntactic Task Analysis. BlackboxNLP 2018, pp. 359–361. DOI 10.18653/v1/W18-5448.
45. A. Conneau, G. Kruszewski, G. Lample, L. Barrault, M. Baroni. What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties. ACL 2018, pp. 2126–2136. DOI 10.18653/v1/P18-1198.
46. T. Chen, S. Kornblith, M. Norouzi, G. Hinton. A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020, PMLR 119:1597–1607.
47. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick. Masked Autoencoders Are Scalable Vision Learners. CVPR 2022, pp. 16000–16009.
48. C. Ying et al. Do Transformers Really Perform Badly for Graph Representation? NeurIPS 2021. arXiv:2106.05234.

**Multi-task 權重與資料混合**

49. A. Kendall, Y. Gal, R. Cipolla. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR 2018, pp. 7482–7491. DOI 10.1109/CVPR.2018.00781.
50. Z. Chen, V. Badrinarayanan, C.-Y. Lee, A. Rabinovich. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML 2018, PMLR 80:794–803.
51. G. Lample, A. Conneau. Cross-lingual Language Model Pretraining. arXiv:1901.07291, 2019.
52. A. Conneau et al. Unsupervised Cross-lingual Representation Learning at Scale. ACL 2020, pp. 8440–8451. DOI 10.18653/v1/2020.acl-main.747.
53. N. Arivazhagan et al. Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges. arXiv:1907.05019, 2019.
54. Octo Model Team. Octo: An Open-Source Generalist Robot Policy. RSS 2024. DOI 10.15607/RSS.2024.XX.090.

**Masked modeling**

55. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019, pp. 4171–4186. DOI 10.18653/v1/N19-1423.
56. P. Wu et al. Masked Trajectory Models for Prediction, Representation, and Control. ICML 2023, PMLR 202:37607–37623.
57. F. Liu, H. Liu, A. Grover, P. Abbeel. Masked Autoencoding for Scalable and Generalizable Decision Making. NeurIPS 2022. arXiv:2211.12740.

**Adaptation**

58. Y. Sun et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. ICML 2020, PMLR 119:9229–9248.
59. N. Hansen et al. Self-Supervised Policy Adaptation during Deployment. ICLR 2021. arXiv:2007.04309.
60. E. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.
61. X. Li, Y. Grandvalet, F. Davoine. Explicit Inductive Bias for Transfer Learning with Convolutional Networks. ICML 2018, PMLR 80:2825–2834.
62. Z. Li, D. Hoiem. Learning without Forgetting. ECCV 2016; IEEE TPAMI 40(12):2935–2947, 2018. DOI 10.1109/TPAMI.2017.2773081.

**最佳化與評估**

63. I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019. arXiv:1711.05101.
64. P. Henderson et al. Deep Reinforcement Learning that Matters. AAAI 2018. DOI 10.1609/aaai.v32i1.11694.
65. R. Agarwal et al. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021. arXiv:2108.13264.
66. J. Kaplan et al. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020.
