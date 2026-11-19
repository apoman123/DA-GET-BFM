# 附錄:實作對照(branch `feature/proposal-v4`)

本附錄記錄 proposal v4 每一項在 repo 中的實作位置、設定方式與測試。所有選項**預設關閉**,原本的模型與 checkpoint 行為不變;英文的完整說明在 `docs/zero_shot.md`。

**重要限制**:開發機沒有 GPU,因此**沒有跑任何大規模訓練或 ablation**。所有 [H] 項目都還沒有被證實或否定。目前的驗證是單元測試,加上 CPU 上的端到端 smoke run(極小模型、幾個 step)。

## A. 實作對照表

| 項目 | 實作 | 設定 | 測試 |
|---|---|---|---|
| A1 physics normalization | `model/training/dataset.py`(τ_peak/M_ii、g、g/L、Froude 單位) | data config `target_normalization: "physics"` | `tests/test_zero_shot.py` |
| A1 M-norm loss | `model/pretraining.py` `gauss_metric_error`(正規化質量矩陣) | `acceleration_metric: "mass"` | 同上 |
| A2 統一 50 Hz | `model/training/resampling.py`;模擬 source 直接以 `control_dt` 模擬 | `control_dt: 0.02` | 同上 |
| A3 無因次化 | `physics_schema.py`(每個特徵的 M/L/T 指數)、`modeling_da_get.py`(`physics_units`)、`physics.py`(`froude_scale_base_features`)、RL observer | `dimensionless_physics`、`dimensionless_inputs` | 同上(質量縮放不變性) |
| B0 量測 | `model/evaluation/probes.py`、`physics_metrics.py`、`counterfactual.py`、`scripts/diagnose_physics_location.py` | CLI | `tests/test_evaluation.py` |
| B1 結構化介面 | `pretraining.py` `DaGetPhysicsInterface`(Ĝ = WWᵀ + diag、f̂、K̂);operator 與 consistency loss;rollout effect | `head_type`、`interface_rank`、`effect_mode` | `tests/test_zero_shot.py`(真 G 代入時 R/R⁻¹ 讀出等於 target) |
| B2 actor 介面 bias | `modeling_bfm.py`(`interface_operator`、actor 每層 bias) | `BfmConfig.use_interface_bias` | 同上 |
| B3 heads lr | 既有 `FbCprConfig.lr_physics_heads` | config | — |
| C1 root token | `physics.py` `root_features`、encoder memory 加 1 個 token | `use_root_token` | 同上 |
| C2 接觸 head | `episodes.py`(`contact_label_valid`)、`dataset.py`(link→joint)、`pretraining.py` | `use_contact_head` | 同上 |
| C3 centroidal | `model/centroidal.py`(自由落體時 ṗ = mg、k̇_G = 0 已驗證) | `use_centroidal_head` | 同上 |
| D1 形態生成器 | `model/morphology.py`、`model/training/procedural.py`、`scripts/generate_morphologies.py` | source `type: "procedural"` | `tests/test_morphology.py` |
| D2 抽樣平衡、單次 encode | `model/training/sampling.py`、`trainer.py`(WeightedRandomSampler)、`group_variants` | `sampling_alpha`、`sampling_group`、`group_variants` | `tests/test_zero_shot.py`(分組前後各任務 loss 相同) |
| D3 鏡像 | `model/training/symmetry.py`(由幾何推導 mirror map) | source `mirror: true` | `tests/test_symmetry.py` |
| D4 拓撲 bias | `modeling_da_get.py` `DaGetMorphologyModulation` | `topology_bias: get / none / modulation` | `tests/test_zero_shot.py`(初始化等同原 encoder) |
| E1 history、sysID head | data config `history_size: 25`;`use_sysid_head` | 同左 | 同上 |
| E2 規格隨機化 | `dataset.py` `spec_noise`(範圍取自 repo 的 `DEFAULT_RANDOMIZATION`) | `spec_noise` | 同上 |
| E3 masked physics modeling | encoder mask embedding、`pretraining.py` mask heads | `physics_mask_ratio`(0.15 / 0.75) | 同上 |
| F1 TTA | `model/adaptation.py`(LoRA、L2-SP、LwF)、`scripts/adapt_da_get.py` | CLI | `tests/test_adaptation.py` |
| G1 action / gain | `rl/trainer.py` `action_scale`(依 family)、`ground_sim.py` `gain_mode: "mass"` | RL data config | `tests/test_zero_shot.py` |
| G2 程序化形態 RL | 每台機器人自帶站姿與腳幾何(`record.metadata["env"]`) | `configs/rl/zero_shot_procedural.json` | `tests/test_morphology.py` |
| G3 retargeting | `model/rl/retarget.py`、`scripts/retarget_motion.py` | CLI | `tests/test_retarget.py` |
| R6 修正 | `StateView.mass`、replay buffer、`encode_state`、deployment | 預設啟用 | `tests/test_zero_shot.py` |
| 評估 L0–L4 | `configs/splits/zero_shot.json`、`model/evaluation/statistics.py`(IQM、stratified bootstrap,以機器人為單位)、`scripts/evaluate_zero_shot.py` | CLI | `tests/test_evaluation.py` |

## B. 設定檔

- `configs/data/zero_shot.json`:50 Hz、0.5 s history、physics normalization、無因次輸入、spec noise、grouped variants、family temperature sampling(α = 0.3)。訓練機器人為 Go1、ANYmal D、G1,加上程序化四足與人形(各 100 台)。H1/Go2/Talos(L4)、六足(L3)與 L2 拓撲都排除。
- `configs/pretraining/zero_shot.json`:`scripts/train_physics_da_get.py` 的完整參數(lr 5e-4、warmup 100、batch 64、2000 steps,取自 `docs/training.md`)。
- `configs/rl/zero_shot_procedural.json`:真實與程序化機器人在 MuJoCo 地面上的 RL。
- `tests/test_zero_shot.py` 會用 split manifest 檢查,保證上述設定不含任何 held-out 機器人。

## C. 端到端 smoke run(CPU)

把 `configs/data/zero_shot.json` 縮小(每個 source 1 個 episode、30 步、程序化每類 2 台)後執行 `scripts/train_physics_da_get.py`。所有選項同時開啟:`linear_residual`、M-norm、rollout effect、ModuMorph modulation、root token、無因次化、contact、centroidal、sysID、masking、spec noise、grouped variants、temperature sampling。

結果:3 個 step 訓練與評估都完成,所有 loss 都是有限值。例如 eval loss_operator 3.18、consistency 234、effect_rollout 14.9、contact 0.70、centroidal 0.070、sysid 0.063。

初始化時 consistency loss 偏大,原因是 K̂ 尚未對齊 Ĝ⁺。loss 權重沿用 repo 預設值 1;是否改用 uncertainty weighting 或 GradNorm,要由實際訓練決定。

## D. 實作時發現的文獻細節更正

- **LoRA**:Hu et al. Table 6 掃的是 r ∈ {1, 2, 4, 8, 64},不是 1–8。論文沒有給 A 的初始化標準差,作者程式碼用 `kaiming_uniform_`。
- **GenBot-1K**(Ai et al. Table 6):變化參數是離散集合(thigh/calf/torso ∈ {0.4, 0.8, 1.0, 1.2, 1.6}、knee limit ∈ {0.2, 0.6, 1.0});作者的 generator 一次只改一個幾何因子。論文與程式碼不一致之處,`model/morphology.py` 的 docstring 註明採用哪一個。
- **GenLoco**:400–1200 kg/m³ 等於 A1 密度 × [0.5, 1.5],預設對每個 link 套用這個相對規則。
- **Peng et al. 2020**:論文沒有寫尺寸比例,程式碼用手設常數。因此自動的特徵長度比例標為 [H]。
- **L2-SP**:α 由交叉驗證決定(設為必要參數);β = 0.01 是論文報告「一致被選到」的值。
- **Talos**:URDF 的 `mechanicalReduction` 是負值(反向裝的 actuator)。parser 改為取絕對值,Talos 因此可以作為 L4 機器人。
- **真實機器人的 URDF 質量與限位並不左右對稱**(Go1、H1 差約 1%,G1 約 3%,ANYmal D 約 10%),所以 D3 鏡像只用在對稱的程序化機器人上。

## E. 尚未完成、需要 GPU 或額外資料的部分

1. Phase 0 的診斷(B0)要在**實際訓練好的 checkpoint** 上跑,才能依決策規則判斷 B1 是否必要。
2. 第 5.6 節的所有 ablation、形態數 scaling(5% → 100%)、多 seed 與 bootstrap CI,都需要 GPU 長時間訓練。
3. 程序化機器人、Go2 與 Talos 都沒有 expert motion,所以 BFM 的 L1–L4 評估需要先用 G3 retargeting 產生 motions;`scripts/evaluate_zero_shot.py` 會把缺資料的機器人列為 missing,不會編造數字。
4. L1(參數變化)還缺一個工具來寫出修改後的 URDF;LocoMuJoCo 的其他人形只有 MJCF,需要先轉成 URDF。
5. Isaac Lab 路徑仍只用 fake 測試過(原本的限制)。
