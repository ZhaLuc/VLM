# Human input required now

S6 decision recorded by the human researcher: `APPROVE`.

Clip: mac_king_s006 / Movie6.MP4
Question: Which hand contains the coin after the apparent transfer?
Ground truth: right
Human approval: APPROVED

Clip: mac_king_s007 / Movie7.MP4 — leave pending.
Candidate question: Which hand contains the coin after the transfer?
Candidate answer: left
APPROVE / EDIT / REJECT

Do not gold-label Wikimedia clips. Do not splice S1/S2 onto S6/S7.
Approved gold: 1 of 5.

Remaining first-baseline blockers:

DATA: S6 approved and benchmark-eligible. 4 more clips for a 5-clip pilot (later).
ENVIRONMENT: CUDA-enabled PyTorch and NVIDIA GPU required.
MODEL: Local Qwen2.5-VL-3B-Instruct or 7B-Instruct weights required.
