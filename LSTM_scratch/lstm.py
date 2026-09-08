import torch
from tqdm import tqdm

# Set CUDA default device if available


# 1. Parse Data with UTF-8 Encoding (Handles math symbols and unicode)
data = open("physics_textbook.txt", "r", encoding="utf-8", errors="ignore").read()

chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


# 2. Hyperparameters
hidden_size = 256
seq_size = 100      # 100-character window for long context
learning_rate = 1e-3

# 3. Model Parameters Initialization (Kaiming / Xavier Scaling)
W_x = torch.randn(4 * hidden_size, vocab_size) * (2.0 / hidden_size) ** 0.5
W_h = torch.randn(4 * hidden_size, hidden_size) * (2.0 / hidden_size) ** 0.5
b_gate = torch.zeros((4 * hidden_size, 1))

Why = torch.randn(vocab_size, hidden_size) * (2.0 / vocab_size) ** 0.5
by = torch.zeros((vocab_size, 1))

# Custom Softmax with Temperature Control
def softmax(y, temperature=1.0):
    y_scaled = y / max(temperature, 1e-8)
    exp_y = torch.exp(y_scaled - torch.max(y_scaled))
    return exp_y / torch.sum(exp_y)

# 4. GPU-Optimized Forward + Backward Pass (Manual BPTT)
def lstm_lossFunc(inputs, targets, hprev, cprev):
    cs, hs, ps, ys = {}, {}, {}, {}
    f_gates, i_gates, c_tilde, o_gates = {}, {}, {}, {}

    hs[-1] = torch.clone(hprev)
    cs[-1] = torch.clone(cprev)
    loss = torch.tensor(0.0)  # Accumulated purely as a CUDA tensor

    # --- Forward Pass ---
    for t in range(len(inputs)):
        # Linear projection for all 4 gates via direct column lookup
        gates_raw = W_x[:, inputs[t] : inputs[t] + 1] + W_h @ hs[t - 1] + b_gate

        f_raw = gates_raw[0 * hidden_size : 1 * hidden_size]
        i_raw = gates_raw[1 * hidden_size : 2 * hidden_size]
        c_raw = gates_raw[2 * hidden_size : 3 * hidden_size]
        o_raw = gates_raw[3 * hidden_size : 4 * hidden_size]

        f_gates[t] = torch.sigmoid(f_raw)
        i_gates[t] = torch.sigmoid(i_raw)
        c_tilde[t] = torch.tanh(c_raw)
        o_gates[t] = torch.sigmoid(o_raw)

        cs[t] = f_gates[t] * cs[t - 1] + i_gates[t] * c_tilde[t]
        hs[t] = o_gates[t] * torch.tanh(cs[t])

        ys[t] = Why @ hs[t] + by
        ps[t] = softmax(ys[t], temperature=1.0)

        prob = torch.clamp(ps[t][targets[t], 0], min=1e-12)
        loss = loss - torch.log(prob)  

    # --- Backward Pass ---
    dW_x = torch.zeros_like(W_x)
    dW_h = torch.zeros_like(W_h)
    db_gate = torch.zeros_like(b_gate)
    dWhy = torch.zeros_like(Why)
    dby = torch.zeros_like(by)

    dhnext = torch.zeros_like(hs[0])
    dcnext = torch.zeros_like(cs[0])

    for t in reversed(range(len(inputs))):
        dy = torch.clone(ps[t])
        dy[targets[t]] -= 1

        dWhy += dy @ hs[t].T
        dby += dy

        dh = Why.T @ dy + dhnext
        tanh_cs = torch.tanh(cs[t])

        do_raw = dh * tanh_cs * o_gates[t] * (1 - o_gates[t])
        dc = dh * (1 - tanh_cs**2) * o_gates[t] + dcnext

        df_raw = dc * cs[t - 1] * f_gates[t] * (1 - f_gates[t])
        di_raw = dc * c_tilde[t] * i_gates[t] * (1 - i_gates[t])
        dc_tilde_raw = dc * i_gates[t] * (1 - c_tilde[t] ** 2)

        dcnext = dc * f_gates[t]

        dgates_raw = torch.cat([df_raw, di_raw, dc_tilde_raw, do_raw], dim=0)

        db_gate += dgates_raw
        dW_h += dgates_raw @ hs[t - 1].T
        dW_x[:, inputs[t] : inputs[t] + 1] += dgates_raw

        dhnext = W_h.T @ dgates_raw

    # Normalize gradients and clip
    for dparam in [dW_x, dW_h, dWhy, db_gate, dby]:
        dparam /= len(inputs)
        dparam.clamp_(-5, 5)

    
    return (
        (loss.item() / len(inputs)),
        dW_x,
        dW_h,
        dWhy,
        db_gate,
        dby,
        hs[len(inputs) - 1],
        cs[len(inputs) - 1],
    )

# 5. Character-Level Temperature Sampling
def sample(h, c, seed_ix, n, temperature=0.7):
    ixes = [seed_ix]

    with torch.no_grad():
        for _ in range(n):
            gates_raw = W_x[:, seed_ix : seed_ix + 1] + W_h @ h + b_gate

            f = torch.sigmoid(gates_raw[0 * hidden_size : 1 * hidden_size])
            i = torch.sigmoid(gates_raw[1 * hidden_size : 2 * hidden_size])
            c_tilde = torch.tanh(gates_raw[2 * hidden_size : 3 * hidden_size])
            o = torch.sigmoid(gates_raw[3 * hidden_size : 4 * hidden_size])

            c = f * c + i * c_tilde
            h = o * torch.tanh(c)

            y = Why @ h + by
            p = softmax(y, temperature=temperature).ravel()

            seed_ix = torch.multinomial(p, 1).item()
            ixes.append(seed_ix)

    return "".join(idx_to_char[i] for i in ixes)

# Checkpoint Function
def save_checkpoint(filepath="physics_char_model.pt"):
    checkpoint = {
        "W_x": W_x,
        "W_h": W_h,
        "Why": Why,
        "b_gate": b_gate,
        "by": by,
        "hidden_size": hidden_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }
    torch.save(checkpoint, filepath)
    print(f"\n[Checkpoint Saved: {filepath}]")

# 6. Adam Optimizer Memory Allocation
mW_x, vW_x = torch.zeros_like(W_x), torch.zeros_like(W_x)
mW_h, vW_h = torch.zeros_like(W_h), torch.zeros_like(W_h)
mWhy, vWhy = torch.zeros_like(Why), torch.zeros_like(Why)
mb_gate, vb_gate = torch.zeros_like(b_gate), torch.zeros_like(b_gate)
mby, vby = torch.zeros_like(by), torch.zeros_like(by)

# 7. Training Loop
n, p = 0, 0
smooth_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()
p_bar = tqdm(range(50000), desc="Step = 0", colour="red")
beta1, beta2 = 0.9, 0.999
t = 0

for epoch_idx in p_bar:
    # Reset hidden and cell states at epoch start or end of text
    if p + seq_size + 1 >= len(data) or n == 0:
        hprev = torch.zeros((hidden_size, 1))
        cprev = torch.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_idx[ch] for ch in data[p : p + seq_size]]
    targets = [char_to_idx[ch] for ch in data[p + 1 : p + seq_size + 1]]

    # Sample output every 1000 steps
    if epoch_idx > 0 and epoch_idx % 1000 == 0:
        sample_text = sample(hprev, cprev, inputs[0], n=300, temperature=0.7)
        print(f"\n--- Sample Step {epoch_idx} (Temp 0.7) ---\n{sample_text}\n--------------------")

    # Save checkpoint every 10,000 steps
    if epoch_idx > 0 and epoch_idx % 10000 == 0:
        save_checkpoint(f"physics_char_step_{epoch_idx}.pt")

    loss, dW_x, dW_h, dWhy, db_gate, dby, hprev, cprev = lstm_lossFunc(
        inputs, targets, hprev, cprev
    )
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    p_bar.set_description(f"Step {epoch_idx}")
    p_bar.set_postfix(loss=f"{smooth_loss:.4f}")
    t += 1

    # In-place Adam Parameter Update
    with torch.no_grad():
        for param, dparam, m_mems, v_mems in zip(
            [W_x, W_h, Why, b_gate, by],
            [dW_x, dW_h, dWhy, db_gate, dby],
            [mW_x, mW_h, mWhy, mb_gate, mby],
            [vW_x, vW_h, vWhy, vb_gate, vby],
        ):
            m_mems.mul_(beta1).add_(dparam, alpha=1 - beta1)
            v_mems.mul_(beta2).add_(dparam**2, alpha=1 - beta2)

            m_corrected = m_mems / (1 - beta1**t)
            v_corrected = v_mems / (1 - beta2**t)

            step = m_corrected / (torch.sqrt(v_corrected) + 1e-8)
            param.sub_(step, alpha=learning_rate)

    p += seq_size
    n += 1

# Save Final Trained Weights
save_checkpoint("physics_char_final.pt")