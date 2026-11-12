# CardioMind-FEMO: Algorithm Documentation

## 1. CardioMind-FEMO — Federated Multi-Objective Optimization

### 1.1 Problem Formulation

Standard federated learning optimizes a single prediction objective across distributed clients. CardioMind-FEMO extends this to simultaneously optimize four objectives:
1. **Primary Binary Classification** ($\mathcal{L}_{\text{pred}}$)
2. **Probability Calibration** ($\mathcal{L}_{\text{cal}}$)
3. **Proximal Parameter Drift Control** ($\mathcal{L}_{\text{prox}}$)
4. **Bounded Personalized Risk Consistency** ($\mathcal{L}_{\text{risk}}$)

**Global objective**: Find model parameters $\theta^*$ that minimize the federated multi-objective loss across all $K$ clients while balancing stability and generalization under non-IID data distributions.

---

### 1.2 Multi-Objective Local Loss Formulation

For client $k$ with local dataset $\mathcal{D}_k$ of $n_k$ samples:

$$\mathcal{L}_{\text{FEMO}}(\theta_k; \theta_t) = \tilde{\mathcal{L}}_{\text{pred}}(\theta_k) + \lambda_c(t) \cdot \tilde{\mathcal{L}}_{\text{cal}}(\theta_k) + \lambda_p(t) \cdot \tilde{\mathcal{L}}_{\text{prox}}(\theta_k; \theta_t) + \lambda_r(t) \cdot \tilde{\mathcal{L}}_{\text{risk}}(\theta_k)$$

Where each component $\tilde{\mathcal{L}}_j$ is normalized by its running Exponential Moving Average (EMA) magnitude:
$$\tilde{\mathcal{L}}_j = \frac{\mathcal{L}_j}{\text{EMA}(\mathcal{L}_j) + \epsilon}$$

#### 1. $\mathcal{L}_{\text{pred}}$ — Prediction Loss (Binary Cross-Entropy)
$$\mathcal{L}_{\text{pred}} = -\frac{1}{n_k} \sum_{i=1}^{n_k} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$
- $y_i \in \{0, 1\}$: ground-truth cardiovascular event label
- $p_i \in (0, 1)$: predicted probability output from the neural tabular model

#### 2. $\mathcal{L}_{\text{cal}}$ — Calibration Loss (Brier-style)
$$\mathcal{L}_{\text{cal}} = \frac{1}{n_k} \sum_{i=1}^{n_k} (p_i - y_i)^2$$
- Penalizes miscalibrated confidence scores and minimizes Expected Calibration Error (ECE)

#### 3. $\mathcal{L}_{\text{prox}}$ — Proximal Regularization (Client Drift Control)
$$\mathcal{L}_{\text{prox}} = \frac{1}{2} \|\theta_k - \theta_t\|^2$$
- Constrains local client models from diverging from global broadcast parameters $\theta_t$

#### 4. $\mathcal{L}_{\text{risk}}$ — Bounded Huber Risk-Consistency Loss
$$\mathcal{L}_{\text{risk}} = \frac{1}{n_k} \sum_{i=1}^{n_k} \rho_\delta \left( r_{i,t} - \hat{r}_{i,t-1} \right)$$
where $\rho_\delta(z)$ is the Huber loss with threshold $\delta = 0.1$:
$$\rho_\delta(z) = \begin{cases} \frac{1}{2} z^2 & \text{if } |z| \le \delta \\ \delta \left(|z| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}$$
- Prevents extreme risk score fluctuations from dominating backpropagation gradients
- $\hat{r}_{i,t-1}$: prior model risk estimate on patient $i$

---

### 1.3 Progressive Multi-Objective Scheduling

To allow the representation layers to converge on primary feature patterns before regularizing:
- $\lambda_p(t) = \lambda_{p,\text{base}} \cdot \min\left(1.0, \frac{t}{T_{\text{warmup}}}\right)$
- $\lambda_c(t) = \lambda_{c,\text{max}} \cdot \left(\frac{t}{T}\right)$
- $\lambda_r(t) = \lambda_{r,\text{max}} \cdot \left(\frac{t}{T}\right)$

Default Hyperparameters:
- $\lambda_{c,\text{max}} = 0.05$
- $\lambda_{p,\text{base}} = 0.01$
- $\lambda_{r,\text{max}} = 0.01$
- Huber $\delta = 0.1$, EMA decay $\beta = 0.90$

---

### 1.4 Adaptive Reliability & Stability Aggregation

Server aggregates client parameter updates using bounded reliability and stability weighting:
$$q_k' = \text{clip}\left(\text{AUC}_k + (0.25 - \text{Brier}_k), 0.0, 1.0\right)$$
$$s_k' = \text{clip}\left(\frac{\Delta \theta_k^{(t)} \cdot \Delta \theta_k^{(t-1)}}{\|\Delta \theta_k^{(t)}\| \|\Delta \theta_k^{(t-1)}\| + \epsilon}, 0.0, 1.0\right)$$
$$u_k = n_k \cdot (1 + \eta q_k') \cdot (1 + \mu s_k')$$
$$w_k = \text{clip}\left(\frac{u_k}{\sum_j u_j}, w_{\text{min}}, w_{\text{max}}\right) \quad \text{renormalized so } \sum_k w_k = 1$$
$$\theta_{t+1} = \sum_{k=1}^K w_k \cdot \theta_k$$

---

### 1.5 Server-Side Validation & Checkpoint Selection

After each round $t \in [1, T]$, the server evaluates the global model on the global validation set $(X_{\text{val}}, y_{\text{val}})$:
$$S_{\text{val}}(t) = \text{AUC}_{\text{val}}(t) - 0.5 \cdot \text{Brier}_{\text{val}}(t) - 0.5 \cdot \text{ECE}_{\text{val}}(t)$$
$$\theta^* = \theta_{t^*}, \quad t^* = \arg\max_t S_{\text{val}}(t)$$

The locked test set $(X_{\text{test}}, y_{\text{test}})$ is evaluated ONLY ONCE at the end on $\theta^*$.

---

### 1.6 APCRS — Adaptive Personalized Cardiovascular Risk Score

$$\text{APCRS}_t = 100 \times \text{clip}\left(\alpha \cdot p_t + \beta \cdot R_t + \gamma \cdot \Delta R_t, 0, 1\right)$$
where $\alpha + \beta + \gamma = 1$.

| APCRS Range | Category | Action / Follow-up |
|---|---|---|
| 0–29 | Low | Routine lifestyle maintenance |
| 30–59 | Moderate | Dietary modification, moderate activity tracking |
| 60–79 | High | Clinical consult, intensive BP / lipid monitoring |
| 80–100 | Very High | Urgent clinical intervention |
