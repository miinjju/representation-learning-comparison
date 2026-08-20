# 표현학습 방식 비교 — MAE vs VAE

자기지도 표현학습 방식에 따라 표현 품질이 어떻게 달라지는지
동일 조건에서 통제 비교했습니다. 평가는 Linear Probing으로 했습니다.

## 결과

| 모델 | Linear Probing Acc |
|---|---|
| **MAE** | **63.00%** |
| VAE | 41.16% |
| VAE (backbone → ResNet) | 41.51% |

## 통제 조건

두 모델의 구조 차이 외에 다른 변수가 섞이지 않도록 맞췄습니다.

| 항목 | 값 |
|---|---|
| 데이터셋 | CIFAR-10 (50k / 10k 표준 split) |
| Transform | ToTensor()만 — 정규화·증강 없음 |
| Feature dim | 256 |
| Batch size | 64 |
| Optimizer | Adam, lr 1e-3 |
| Pretrain | 50 epoch |
| Linear probe | 50 epoch (encoder freeze) |

## 모델 구성

**MAE** — 이미지를 패치로 나눠 일부를 가리고 복원하며 학습
- Patch size 4 (32×32 → 8×8 = 64 패치), mask ratio 0.75
- Encoder: 4-layer Transformer (dim 256, head 8)
- Decoder: 2-layer Transformer (dim 128, head 4)
- Loss: 마스킹된 패치에만 MSE
- Linear probe 시에는 마스킹 없이 전체 패치를 쓰고, Global Average Pooling으로 특징 추출

**VAE** — 잠재 분포를 학습하며 이미지를 복원
- CNN Encoder: Conv 3→32→64→128→256 (stride 2, BatchNorm + ReLU)
- latent_dim 256, `fc_mu` / `fc_logvar`
- Linear probe는 `mu`를 특징으로 사용

## 가설 검증

두 모델의 격차가 backbone 표현력 차이 때문인지 확인하려고,
VAE의 backbone을 ResNet으로 바꿔 다시 실험했습니다.
41.51%로 0.35%p 오르는 데 그쳐, 원인이 backbone이 아니라
**잠재공간 압축과 KL 규제**에 있다고 보았습니다.

## 한계

- KL 규제 강도를 직접 바꿔보는 ablation은 하지 못했습니다
- DDPM은 구현과 리뷰에 그쳐 정량 비교에는 넣지 못했습니다
- CIFAR-10 단일 데이터셋, 50 epoch로 제한된 조건입니다

## 파일

- `mae_cifar10.ipynb` — MAE 사전학습 · Linear Probing · 시각화
- `vae_cifar10.ipynb` — VAE 사전학습 · Linear Probing · 시각화

각 노트북에 복원 이미지, Loss 곡선, Confusion Matrix 시각화가 포함되어 있습니다.
학습된 체크포인트는 포함하지 않았습니다.
