---
layout: post
title:  "블로그 시작하기"
date:   2025-07-15 12:00:00 +0900
categories: 딥러닝
use_math: true
---

안녕하세요! 이 블로그에서는 딥러닝과 수학에 대한 내용을 정리해나갈 예정입니다.

## Mean Field Approximation (변분 추론)

변분 추론에서 Mean Field Approximation은 복잡한 사후 분포를 독립적인 인수들의 곱으로 근사하는 방법입니다.

### 기본 개념

사후 분포 $p(\mathbf{z}|\mathbf{x})$를 다음과 같이 근사합니다:

$$q(\mathbf{z}) = \prod_{i=1}^{M} q_i(z_i)$$

여기서 $\mathbf{z} = \{z_1, z_2, \ldots, z_M\}$는 잠재 변수들입니다.

### KL Divergence 최소화

목표는 KL divergence를 최소화하는 것입니다:

$$\text{KL}(q(\mathbf{z}) \| p(\mathbf{z}|\mathbf{x})) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

### Mean Field 업데이트 공식

각 인수 $q_j(z_j)$에 대한 최적해는 다음과 같습니다:

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{z}, \mathbf{x})] + \text{const}$$

여기서 $q_{-j} = \prod_{i \neq j} q_i(z_i)$이고, $\mathbb{E}_{q_{-j}}$는 $z_j$를 제외한 모든 변수에 대한 기댓값입니다.

### Evidence Lower Bound (ELBO)

변분 추론의 목적 함수는 다음과 같은 ELBO를 최대화하는 것입니다:

$$\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z})]$$

이는 다음과 같이 쓸 수도 있습니다:

$$\log p(\mathbf{x}) = \mathcal{L}(q) + \text{KL}(q(\mathbf{z}) \| p(\mathbf{z}|\mathbf{x}))$$

따라서 ELBO를 최대화하면 KL divergence를 최소화하게 됩니다.