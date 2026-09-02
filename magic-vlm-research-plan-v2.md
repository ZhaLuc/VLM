# Post-Training a VLM on Magic/Mentalism Demonstrations: Confirmed-Scope Research Plan

Prepared after Professor Zhe Xu's confirmation. This document supersedes the exploratory version from before his reply and is meant to be usable on its own at your next meeting.

**A note on method, per your preferences:** every specific claim about a paper, dataset, framework, or Professor Xu's public research record below was checked against a live source in the last hour, not recalled from memory. Where I could not verify something, I say so. Section 18 separates confirmed facts from inference from open questions - please read that section critically, not just as a formality.

---

## 1. What Professor Xu is actually asking for

Based on his reply as you relayed it, three things are now settled that were previously ambiguous:

1. **The pipeline shape is confirmed correct**: base VLM -> dataset of demonstrations -> model-generated candidate outputs -> human feedback (labels/rankings/critiques/preferences) -> reward or preference model -> post-training (PPO/GRPO/DPO/related) -> evaluation. This matches the general RLHF-for-VLM pattern used in the published literature (Section 3).
2. **Magic and mentalism are the literal domain, not a metaphor.** This rules out the "misdirection is just a stand-in for some other generic capability" framing from before. It also means video content (not just static images) is very likely central, since almost all of the interpretations he's pointing at (method inference, hidden-state inference, misdirection reasoning) depend on motion.
3. **The scientific target is visual reasoning, inference, and explanation from demonstrations** - not physical trick performance, not robot manipulation. This directly rules out interpretations F and G from the earlier document (learning to physically reproduce a trick, or planning robot actions to perform one). Keep robotics out of the first prototype entirely.

His instruction to you was procedural, not just topical: **(a)** survey VLM post-training methods, **(b)** think concretely about what a small prototype dataset of magic/mentalism demonstrations should look like, **(c)** narrow scope based on feasibility, with evaluation - specifically distinguishing genuine reasoning improvement from superficial metric gains - called out as important throughout, not as an afterthought.

---

## 2. Plain-English explanation of the project

A vision-language model (VLM) is a neural network that takes an image or video plus a text prompt and produces text - it can "look" and "talk" at the same time. Right now, if you show a general-purpose VLM a video of a card trick and ask "how was this done," it will probably give you a plausible-sounding but often wrong answer, because it wasn't specifically trained to reason carefully about deception, misdirection, or hidden state - it was trained mostly on cooperative, honestly-presented visual content (cat photos, product images, everyday scenes).

**Post-training** is the general name for anything you do to a model *after* its initial large-scale training to change its behavior in a targeted way, without retraining it from scratch. This project is: take a VLM, show it magic/mentalism demonstrations, get some kind of feedback on how good its answers are (from a human, or from a rule, or both), and use that feedback to nudge the model, via one of several specific algorithms (PPO, GRPO, DPO), toward giving better answers. "Better" has to be defined precisely (Section 6), and you have to be careful that the model doesn't just learn to fool your *grading method* rather than actually getting better at the underlying reasoning (Section 13) - that risk, called reward hacking, is one of the central open problems in this entire subfield, and Professor Xu's emphasis on evaluation suggests he already expects you to take it seriously.

---

## 3. Research landscape

I searched specifically for the categories you listed. Below is the required-field table for the papers most directly relevant to your actual project (method + domain overlap). A longer reference list with additional adjacent papers follows.

| Title | Authors | Venue/Year | URL | Task | Base model | Dataset | Training method | Human prefs? | Reward model? | RL used? | Eval method | Result (as reported) | Relevance to your project |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao, Wang, Zhu, Xu, Song, Zhang, Li, Wu, Guo | arXiv, Feb 2024 | arxiv.org/abs/2402.03300 | Math reasoning | DeepSeekMath-7B (text-only, not a VLM) | Web-scraped math corpus + instruction data | **Introduces GRPO** | No (rule-based reward) | No (rule-based correctness) | Yes | Standard math benchmarks (e.g., MATH) | GRPO improved math reasoning accuracy over the SFT baseline, at lower memory cost than PPO | This is where GRPO itself comes from - not multimodal, but the algorithm you'll likely use |
| Video-R1: Reinforcing Video Reasoning in MLLMs | Feng, Gong, Li, Guo, Wang, Peng, Wu, Zhang, Wang, Yue | arXiv, Mar 2025 | arxiv.org/abs/2503.21776 | Video reasoning (temporal) | Qwen2.5-VL (7B) | Two new video-reasoning datasets built by the authors | **T-GRPO**, a GRPO variant with an explicit temporal-order reward | No (rule-based + a clever temporal-shuffle reward, no human preference labels) | No | Yes | VideoMMMU, MMVU, MVBench, TempCompass, VideoMME | Reported consistent gains over SFT baselines across multiple video-reasoning benchmarks | **The closest existing methodological precedent I found for your project.** T-GRPO rewards the model for doing better on temporally-ordered video than on the same frames shuffled - directly adaptable to magic tricks, where temporal order and misdirection timing are the whole point |
| VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning | (VideoChat-R1 authors) | arXiv, Apr 2025 | arxiv.org/abs/2504.06958 | Spatio-temporal video perception (grounding, tracking) | Qwen2.5-VL (7B) | Existing spatio-temporal QA data | GRPO-based reinforcement fine-tuning (RFT) | No | No (rule-based rewards) | Yes | Temporal grounding, object tracking, VideoMME, MVBench, Perception Test | Large gains on temporal grounding (+31.8) and object tracking (+31.2) with a comparatively small amount of data | Evidence that GRPO-style RFT can be **very data-efficient** for a specific perceptual capability - encouraging for a small student dataset |
| Aligning Large Multimodal Models with Factually Augmented RLHF (LLaVA-RLHF) | Sun, Shen, Zhou, Zhang, Chen, Xie, Yao, Manning, Chen, Finn (approx., based on search results) | arXiv, Sept 2023 | arxiv.org/abs/2309.14525 | Reducing VLM hallucination | LLaVA | Human preference pairs + factual augmentation (captions) | Reward model + **PPO** | Yes | Yes (Bradley-Terry) | Yes | MMHal-Bench (introduced in this paper) | Reduced hallucination relative to baseline LLaVA | The clearest existing example of the *exact* pipeline shape Professor Xu confirmed (dataset -> human prefs -> reward model -> PPO), applied to a different target behavior |
| RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback | Yu, Yao, Zhang, He, Han, Cui, Hu, Liu, Zheng, Sun, et al. | CVPR 2024 (arXiv Dec 2023) | arxiv.org/abs/2312.00849 | Reducing VLM hallucination | LLaVA-based | Human *segment-level corrections* (not just pairwise choices) | Dense DPO variant (DDPO) | Yes | No (uses corrections directly, no separate reward model) | No (DPO is not RL in the sampling sense) | MMHal-Bench and others | Outperformed contemporaneous LLaVA-RLHF using less data | Shows that human *corrections* are a real, well-documented alternative to plain pairwise preferences (relevant to your Dataset E design in Section 6) |
| RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness | Yu, et al. | arXiv, May 2024 | arxiv.org/abs/2405.17220 | Reducing VLM hallucination | Open VLMs | AI-generated (not human) preference data | Iterative DPO | **No - AI feedback instead** | Implicit | No | MMHal-Bench and others | Reported trustworthiness competitive with proprietary models using open-source AI feedback | Directly relevant if you and Professor Xu decide to reduce human-labeling burden via an AI judge (RLAIF, Section 9) |
| VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model | om-ai-lab team | arXiv, Apr 2025 | arxiv.org/abs/2504.07615 | General visual grounding/detection reasoning | Qwen2.5-VL | Task-specific (detection, referring expression) datasets | GRPO, rule-based rewards | No | No | Yes | Task accuracy + out-of-domain generalization tests | Reported substantial gains and better out-of-domain generalization than SFT | A reusable, actively maintained open-source GRPO-for-VLM framework (github.com/om-ai-lab/VLM-R1) you could adapt directly |
| CLEVRER: CoLlision Events for Video REpresentation and Reasoning | Yi, Gan, Li, Kohli, Wu, Torralba, Tenenbaum | ICLR 2020 | arxiv.org/abs/1910.01442 (per standard citation; confirm on arXiv directly) | Video causal reasoning (synthetic) | Various (evaluated many models); introduces neuro-symbolic NS-DR model | Synthetic collision videos | Not RLHF-related - this is an evaluation benchmark, not a training method | No | No | No (evaluation only) | Descriptive/explanatory/predictive/counterfactual QA accuracy | State-of-the-art visual reasoning models at the time performed poorly on explanatory/predictive/counterfactual questions | **Not a training method, but its four-question-type taxonomy (descriptive, explanatory, predictive, counterfactual) is directly reusable as a task schema for your magic-trick dataset** (Section 4/6) |
| CausalVLBench: (benchmarking visual causal reasoning in large vision-language models) | Komanduri et al. | 2025 (per EMNLP 2025 citing source; primary paper not independently confirmed by me) | (found via secondary citation in arXiv:2605.28779) | Visual causal reasoning (causal structure inference, intervention prediction, counterfactual prediction) | Multiple open LVLMs evaluated | Synthetic controlled physical systems | Evaluation only | No | No | No | Multiple-choice accuracy on causal tasks | Found that larger models were needed for effective causal reasoning via in-context learning | Directly relevant task-design template - **I found this only via a secondary citation and could not independently verify all details on the primary source, so treat the specifics with some caution** |
| Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning | Rocamonde, Montesinos, Nava, Perez, Lindner | ICLR 2024 (arXiv Oct 2023) | arxiv.org/abs/2310.12921 | Using a VLM as a reward source for a *separate* RL policy | CLIP-based VLM as reward model; separate MuJoCo policy | N/A (uses text prompts as goal descriptions) | VLM-as-reward, no fine-tuning of the VLM itself | No | The VLM itself acts as the reward, with no separate training | Yes (for the downstream policy) | Task success on MuJoCo control tasks | The VLM-derived reward was competitive with hand-engineered rewards | Relevant if you want a **zero-annotation baseline reward** (e.g., "does this explanation match the trick description text?") before investing in human preference collection |

*Additional adjacent papers, referenced by name and finding rather than full table rows because they are one step further from your exact project:* CounterVQA (2025, arXiv:2511.19923) - a recent video counterfactual-reasoning benchmark reporting that VLMs "achieve reasonable accuracy on simple counterfactual questions" but "performance degrades significantly on complex multi-hop causal chains," directly relevant to how hard your counterfactual-reasoning task (Section 4, capability 8) may be; STVG-R1 (arXiv:2602.11730) and Time-R1, both GRPO variants for spatio-temporal video grounding, corroborating that GRPO-for-video is an active, multi-group research direction, not a one-off; the RLHF-Reward-Modeling GitHub repo (github.com/RLHFlow/RLHF-Reward-Modeling), which contains a working, minimal, from-scratch implementation of both classic Bradley-Terry reward models and generative pairwise preference models, useful as literal starter code for Section 11.

**On magic-specific precedent (your Part 3 question), reconfirmed:** as established previously, I still find no VLM-RLHF paper or public benchmark built specifically around magic tricks or mentalism. The closest prior work remains Zaghi-Lara, Gea, Camí, Martínez, Gómez-Marín (2019), "Playing magic tricks to deep neural networks untangles human deception" (arXiv:1908.07446) - a cognitive-science paper using pose-tracking (DeepLabCut), not a VLM, and with no post-training/RLHF component at all. This absence is good news for novelty (Section 14) and confirms Professor Xu is pointing you at genuinely open territory, not an established subfield with existing benchmarks to beat.

---

## 4. Magic/mentalism task possibilities

Your ten candidate capabilities, cross-referenced against what I could find in the literature, and reframed against CLEVRER's four-question-type taxonomy (descriptive / explanatory / predictive / counterfactual), which is the most directly reusable existing schema I found for exactly this kind of task design:

| # | Capability | Nearest CLEVRER-style category | Existing precedent? | Verdict for a small prototype |
|---|---|---|---|---|
| 1. Trick recognition | Descriptive | General video classification is mature; magic-specific classification is not | Too easy/low-signal on its own; useful only as a pre-processing step |
| 2. Method inference | Explanatory | No magic-specific precedent; general "explain what happened" video QA exists | **Strong candidate** - open-ended, judgeable by preference, matches interpretation B from before |
| 3. Hidden-state inference | Explanatory/descriptive hybrid | Zaghi-Lara et al. (2019) did this with pose-tracking, not a VLM | **Strong candidate** - can have objective, checkable ground truth (rule-based reward, no human labels needed) |
| 4. Temporal reasoning | Descriptive, but requires genuine video (not single-frame) understanding | Video-R1/T-GRPO (2025) built exactly this reward mechanism for general video, not magic | **Directly reusable method** - T-GRPO's shuffle-vs-ordered reward is close to plug-and-play |
| 5. Misdirection reasoning | Explanatory | Theoretically grounded in Kuhn et al.'s misdirection taxonomy (psychology, not ML); no ML implementation found | High novelty, but hard to get ground truth for without magician cooperation |
| 6. Causal reasoning | Explanatory/counterfactual | CLEVRER, CausalVLBench, CounterVQA all test this directly (non-magic domains) | **Strong candidate**, and directly aligned with Professor Xu's own published research focus on causal reinforcement learning (Section 18) |
| 7. Step-by-step explanation | Explanatory | Structurally close to procedure-planning literature (COIN, CrossTask datasets), untried on magic | Good fit, moderate implementation cost |
| 8. Counterfactual reasoning | Counterfactual | CLEVRER and CounterVQA test this on synthetic/general video; CounterVQA specifically reports models degrade sharply on multi-hop causal chains | Interesting but likely the **hardest** to get right for a first prototype - even general-domain models struggle here |
| 9. Procedure reconstruction | Explanatory, sequential | Same procedure-planning literature as #7 | Reasonable, but overlaps heavily with #7 - pick one, not both, for a first pass |
| 10. Mentalism reasoning | Explanatory, often more verbal/psychological than visual | No precedent found; narrower framing of #2 | Plausible narrowing if the professor wants to emphasize the "prediction/forced-choice" side specifically |

**Read on Professor Xu's likely priorities:** given his own published research centers on *temporal* and *causal* reasoning in reinforcement learning (his active NSF CAREER grant is literally titled "Temporal Causal Reinforcement Learning and Control," confirmed via his ASU faculty page - see Section 18), capabilities **4 (temporal reasoning)** and **6 (causal reasoning)**, and by extension **3 (hidden-state inference, which requires both)**, are the ones most likely to connect to his broader research program, not just to the magic-trick project narrowly. This is inference, not confirmed - flag it explicitly at your next meeting rather than assuming it.

---

## 5. Recommended task definition

Given Section 4's analysis, I recommend proposing **two closely related tasks**, framed explicitly in CLEVRER's vocabulary since it is a proven, citable schema and maps cleanly onto both Professor Xu's causal/temporal focus and Professor Xu's own confirmed pipeline:

**Task A - "Explanatory" (method inference, capability 2):** given a short video of a magic/mentalism demonstration, produce a free-text explanation of the most likely hidden mechanism. Judged by preference comparison against a rubric (Section 9).

**Task B - "Descriptive-with-hidden-state" (hidden-state inference, capability 3):** given a video up to a concealment moment plus a targeted question, produce the correct hidden state (which hand, which cup, which card). Judged by exact-match correctness against verifiable ground truth - no human labeling needed for the reward itself.

I recommend **Task B as your literal first experiment** (Section 15) because it has objective, checkable ground truth and is the most directly GRPO-compatible without any preference-data bottleneck, and **Task A as your second stage**, once Task B has validated that the pipeline (data -> model -> reward -> post-training -> eval) actually works end to end. Both tasks share the same underlying video dataset, so building one does not waste effort on the other - this is a genuine two-for-one on data-collection cost.

---

## 6. Recommended prototype dataset

Your five proposed dataset designs, fully specified:

### Dataset A - Trick explanation
- **Exact input:** 10-30 second video clip, single close-up trick, static camera.
- **Exact output:** free-text explanation.
- **Number of examples:** 20-40 clips for a first pass; each clip can generate multiple training signals (see Dataset D below), so raw clip count matters more than raw "example" count.
- **Annotation format:** gold reference explanation (optional, expensive) or pairwise preference between model outputs (cheap, recommended).
- **Annotation cost:** low if pairwise; high if you write gold references for every clip.
- **Video required?** Yes - the "moment" of a sleight-of-hand switch is usually a specific instant, not visible in any single still frame.
- **Train/val/test split:** given the small scale, a simple non-overlapping split by *trick type and performer* (not just by clip) is essential - see Section 13.
- **Metrics:** pairwise win-rate (post-training vs. pre-training), rubric-graded quality.
- **Advantages:** open-ended, directly tests genuine explanatory reasoning, no ground-truth-labeling infrastructure needed if using preferences.
- **Disadvantages:** hard to grade objectively; requires either magic expertise or well-documented "exposed" tricks to know if an explanation is actually correct.
- **Likely bias:** graders (you) may unconsciously favor longer, more confident-sounding, or more jargon-heavy explanations regardless of correctness - a well-documented reward-hacking risk in reward-model literature generally.
- **Likely failure modes:** hallucinated but plausible-sounding wrong mechanisms; over-reliance on superficial visual cues (hand near pocket -> "went in the pocket") without real evidence.

### Dataset B - Hidden-state reasoning
- **Exact input:** video up to the concealment moment + a targeted question ("which cup contains the ball now?").
- **Exact output:** the answer (a location/object identity) + optionally a short justification.
- **Number of examples:** because grading is objective, you can scale faster - 30-50 filmed moments, each turned into multiple question phrasings, could yield a few hundred graded (input, question, answer) triples.
- **Annotation format:** ground truth known in advance (you filmed it, or you know the trick), no subjective human judgment needed for correctness.
- **Annotation cost:** low, once filmed.
- **Video required?** Yes for the input; the answer itself is simple (a label).
- **Train/val/test split:** split by trick and by camera angle (see Section 13's generalization axes) - do not let the same physical setup appear in both train and test.
- **Metrics:** exact-match accuracy, plus accuracy broken out by trick type (some tricks may be systematically harder).
- **Advantages:** objective ground truth means a rule-based reward is possible with zero human preference collection - the fastest path to a first working GRPO run.
- **Disadvantages:** less rich a research question on its own (a single accuracy number is a thinner contribution than a full explanation-quality study).
- **Likely bias:** if hiding locations aren't balanced across your dataset, the model can learn dataset base rates instead of actual reasoning - guard against this explicitly when designing which tricks/angles to include.
- **Likely failure modes:** trivial if the concealment is accidentally visible on camera; impossible (uninformative) if fully hidden with literally no visual cues - camera framing is a real design variable, not an afterthought.

### Dataset C - Temporal causal reasoning
- **Exact input:** video + question ("which action caused the observed change?").
- **Exact output:** action identification (which moment/frame range) + explanation.
- **Number of examples:** similar scale to Dataset B, since this can also have objective ground truth if you know the trick's real mechanism.
- **Video required?** Yes, and this is the design most directly compatible with Video-R1's T-GRPO reward mechanism (Section 3): you can construct a temporally-shuffled version of the same clip as a contrastive negative, rewarding the model for doing substantially better on the correctly-ordered version.
- **Metrics:** localization accuracy (did it point to the right causal moment), plus the shuffle-contrast metric itself as a built-in generalization check.
- **Advantages:** most directly reusable existing method (T-GRPO); most directly aligned with causal-reasoning literature (CLEVRER, CausalVLBench) for eventual comparison.
- **Disadvantages:** requires you to know, and encode, the *actual* causal moment for each clip, which for real magic tricks can be genuinely ambiguous (misdirection often involves multiple simultaneous distracting actions).
- **Likely failure modes:** the model could learn to just point at whichever action happens to be visually salient (a hand moving quickly), rather than the action that is actually causally responsible - exactly the kind of shortcut Video-R1's shuffle-contrast reward was specifically designed to catch.

### Dataset D - Pairwise preference data
- **Exact input:** a video + question, paired with two model-generated candidate answers.
- **Human step:** choose the better of the two.
- **Exact output:** (video, question, response_A, response_B, winner) tuple.
- **Number of examples:** this is generated *from* Datasets A/B/C, not collected separately - for every clip in A, sampling 2-4 candidates and doing pairwise judgments across them yields several preference pairs per clip at low marginal cost.
- **Annotation cost:** the cheapest per-judgment format among all the options here (Section 9 covers why).
- **Advantages:** directly compatible with Bradley-Terry reward modeling (Section 10) and with DPO (Section 8).
- **Disadvantages:** only as good as the rubric behind the judgments; without a clear rubric, "better" collapses into vague taste.
- **Likely failure modes:** annotator (your own) inconsistency across sessions - worth re-checking a sample of your own earlier judgments later.

### Dataset E - Expert correction
- **Exact input:** video + an incorrect model-generated explanation.
- **Human step:** identify the specific error and supply a correction.
- **Exact output:** corrected explanation, ideally localized to the specific wrong claim (this is the RLHF-V pattern from Section 3, "fine-grained correctional feedback").
- **Number of examples:** each is expensive (requires actually diagnosing *why* the model's explanation is wrong), so realistically fewer of these than Dataset D, but each one is higher-signal.
- **Advantages:** richest signal per example; directly usable with dense/segment-level DPO variants.
- **Disadvantages:** requires either real magic expertise or very well-documented trick mechanisms to identify errors confidently; slow.
- **Likely failure modes:** without genuine domain expertise, you risk "correcting" the model with a confident-sounding but actually-wrong correction of your own.

### Recommendation: the smallest dataset that still constitutes a meaningful prototype
**Build Dataset B first** (hidden-state reasoning, 15-25 filmed trick moments, multiple question phrasings each), because it needs no subjective annotation infrastructure and gives you a working rule-based reward almost immediately. **Layer Dataset D on top of the same clips** (pairwise preferences over free-text explanations, sampled from the same footage) as your second, low-marginal-cost addition, which unlocks Dataset A/Task A and Bradley-Terry reward modeling once Task B's pipeline is validated. This gets you two research tasks and two reward types out of a single filming session, which matters a great deal given your realistic time budget.

---

## 7. VLM candidates

Reconfirmed and updated with video-reasoning-specific evidence:

| Model | Params | Video support | Open weights | RL/post-training tooling | Notable corroboration |
|---|---|---|---|---|---|
| **Qwen2.5-VL-3B/7B-Instruct** | 3.8B / ~8B | Dynamic-resolution, dynamic-frame-rate video understanding (per the official Qwen2.5-VL description) | Yes | TRL's official `grpo_vlm.py` example; `2U1/Qwen-VL-Series-Finetune` repo (LoRA/QLoRA, video training, DPO, GRPO); verl and OpenRLHF both explicitly list Qwen2.5-VL support | **This is literally the base model used in Video-R1 and VideoChat-R1** (Section 3), i.e., the two closest existing precedents for exactly your kind of video-temporal-reasoning GRPO project both independently chose it |
| **Qwen3-VL** (smaller dense variants) | Varies, smaller dense variants available | Improved long-video temporal grounding (interleaved-MRoPE, per the official repo) | Yes (Apache 2.0) | Same finetune repo added Qwen3-VL support | Newest generation; worth considering if you want the latest temporal-grounding improvements specifically |
| **InternVL3/3.5** | Multiple sizes | Yes | Yes | Less directly confirmed RL-specific tooling in my searches versus Qwen | Not independently verified for your exact workflow - would need a follow-up check |
| **LLaVA family** | 7B+ | Limited/variable | Yes | Used in LLaVA-RLHF, RLHF-V, RLAIF-V (Section 3) - meaning the most preference-learning-specific prior art exists for this family | Generally heavier than Qwen2.5-VL-3B; strong prior-art fit but possibly less actively maintained tooling today |

**Recommendation, updated:** **Qwen2.5-VL-7B-Instruct** (stepping up from the 3B recommendation in the earlier document) given that both Video-R1 and VideoChat-R1 - your two closest methodological precedents - independently standardized on the 7B variant for genuine video-temporal-reasoning work, suggesting the 3B model may be undersized specifically for the temporal/causal reasoning capabilities Section 4/5 identified as central. If compute becomes a real constraint, falling back to 3B with LoRA remains a reasonable compromise, but I would not assume 3B is sufficient for temporal/causal video reasoning specifically without checking - this is a case where I'm updating my prior recommendation based on new evidence rather than repeating the earlier guess.

**Image vs. video, short vs. long video, frames-plus-text, tradeoffs:**
- **Images alone:** insufficient for capabilities 3-9 in Section 4 (hidden-state, temporal, misdirection, causal, procedural reasoning all require motion). Only adequate for capability 1 (trick-type recognition), which Section 4 already flagged as too shallow a task on its own.
- **Short video (5-30 seconds):** the right scope for a first prototype - matches CLEVRER's clip length convention, matches how close-up magic tricks are naturally structured (a single effect, not a full act), and keeps per-example compute and annotation cost manageable.
- **Long video:** unnecessary complexity for a first pass; relevant only if you later move to a full mentalism *routine* (multiple effects in sequence) rather than a single trick.
- **Frame sequences (sampled stills) vs. true video:** frame sequences are a legitimate cheaper approximation and are literally what Video-R1's T-GRPO reward manipulates directly (shuffling frame order) - so this is not a compromise you'd be inventing yourself, it is how the closest precedent method actually operates internally, which is reassuring.

---

## 8. PPO vs. GRPO vs. DPO

**Level 1 (intuition):**
- **PPO** carefully updates a model's behavior using a separately-trained "critic" network to judge how good each output was compared to expectations.
- **GRPO** does the same job more cheaply by comparing several outputs to the *same* prompt against each other, instead of training a critic.
- **DPO** skips the whole reinforcement-learning loop and directly trains the model on pairs of (preferred, rejected) examples using a specially-derived loss function that behaves, in theory, like RLHF would have.

**Level 2 (formal), with the magic-trick domain now the running example throughout:**

*PPO.* From Schulman et al. (2017, arXiv:1707.06347). For a prompt x (a magic-trick clip + a question) and a sampled response y (the model's answer), PPO maximizes:

```
L_PPO(theta) = E[ min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ]
where ratio = pi_theta(y|x) / pi_theta_old(y|x)
```

`A` (the advantage) requires a separately trained value/critic network that predicts the expected reward for a given prompt, so the algorithm knows whether a specific answer was better or worse than "typical" for that clip. This critic is usually the same size as the policy model, so PPO needs to hold the policy, the critic, a frozen reference model (for a KL penalty preventing the model from drifting too far), and (if used) a reward model all in memory - expensive.

*GRPO.* From DeepSeekMath (Shao et al., 2024, arXiv:2402.03300). For the same prompt x, sample a **group** of G responses, score each with the reward function, and compute the advantage relative to the group's own mean and standard deviation instead of a learned critic:

```
A_i = (r_i - mean(r_1...r_G)) / std(r_1...r_G)
```

**Your requested toy example, worked exactly as specified:** prompt = "watch this trick clip and explain how it was done," G = 4 sampled candidate explanations, with rewards **0.9, 0.7, 0.2, 0.1** (assigned by whatever reward function you've built - Section 11).

- Group mean = (0.9 + 0.7 + 0.2 + 0.1) / 4 = **0.475**
- Group standard deviation (population): deviations from mean are 0.425, 0.225, -0.275, -0.375; squared: 0.1806, 0.0506, 0.0756, 0.1406; mean of squares = 0.1119; sqrt ≈ **0.3345**
- Advantages: A1 = (0.9 - 0.475)/0.3345 ≈ **+1.27**, A2 = (0.7 - 0.475)/0.3345 ≈ **+0.67**, A3 = (0.2 - 0.475)/0.3345 ≈ **-0.82**, A4 = (0.1 - 0.475)/0.3345 ≈ **-1.12**

The training update increases the probability of the first two responses (positive advantage, with the 0.9-reward response pushed hardest) and decreases the probability of the last two (negative advantage, with the 0.1-reward response pushed down hardest). Notice that GRPO used the **relative ranking and spread** of the four scores, not their absolute values - if all four rewards had instead been 0.09, 0.07, 0.02, 0.01 (ten times smaller but same relative pattern), the resulting advantages and thus the training update would be **identical**, because the group-mean/std normalization cancels out the absolute scale. This is exactly why reward *function design* (Section 11) matters more than the RL algorithm choice: GRPO only sees relative differences within a group, so it is only as informative as your reward function's ability to consistently rank multiple candidate answers to the same clip against each other.

*Applying this to multimodal reasoning specifically:* the "action" in a VLM RL setting is not a robot joint torque or a discrete game move (your robotics background's usual RL setting) - it is **an entire generated token sequence** (the whole text answer), and the "trajectory" is the token-by-token generation process, but the reward is typically assigned only once, to the *complete* answer, not per-token (though some newer work, referenced in Section 3's survey citations, does explore denser token-level or step-level rewards). This is the single biggest conceptual difference from robotics RL you already know: there is no physical environment providing step-by-step feedback, no continuous state space, and usually no multi-step episode - it is closer to a one-shot "generate an answer, get a single score" setup repeated across many different prompts, which is part of why GRPO's simplification (no learned value function needed to estimate multi-step returns) works so well here specifically.

*DPO.* Rafailov et al. (2023, arXiv:2305.18290). If y_w is the preferred explanation and y_l is the rejected one (both explaining the same clip):

```
L_DPO(theta) = -E[ log(sigmoid( beta*log(pi_theta(yw|x)/pi_ref(yw|x)) - beta*log(pi_theta(yl|x)/pi_ref(yl|x)) )) ]
```

No sampling loop, no reward model, no critic - this is closer to ordinary supervised fine-tuning with an unusual loss, computed directly from a static dataset of (clip, preferred explanation, rejected explanation) triples (exactly Dataset D from Section 6).

**Level 3 (comparison table, applied to your project):**

| Method | Needs human labels? | Needs a trained reward model? | Compute cost | Best fit given your dataset options |
|---|---|---|---|---|
| PPO | Yes, for the reward model | Yes | Highest | Not recommended as a starting point - too much infrastructure |
| GRPO | Optional - can be rule-based (Dataset B/C) or preference-based (Dataset A/D) | Optional | Medium (needs multi-sample generation per step, ideally with vLLM) | **Best fit for Task B (hidden-state, rule-based reward) as your first RL experiment**, and for Dataset C's T-GRPO-style temporal reward |
| DPO | Yes, as static pairs | No | Lowest | **Best fit for Task A (explanation quality) using Dataset D's preference pairs**, and the cheapest way to validate your pipeline before investing in online RL infrastructure |

**Recommendation for your project specifically:** start with **DPO on Task A** (cheapest, validates the whole data-collection-to-training loop fastest) in parallel with, or immediately followed by, **GRPO on Task B** using a rule-based reward (no annotation bottleneck, and the most directly analogous to Video-R1's T-GRPO approach if you extend it with the temporal-shuffle contrast from Dataset C). Defer PPO entirely unless a specific later need for it emerges - I found no evidence in the current literature that PPO offers an advantage over GRPO for this kind of task that would justify its extra cost for a first prototype.

---

## 9. Human feedback / RLHF / RLAIF

**What "human feedback" would concretely look like in your dataset**, exactly as you asked: a video is shown to the VLM; the VLM produces two candidate explanations (say, at temperature 0.8, so they actually differ); a human (you, to start) reads both against a short rubric (Section 4's misdirection-taxonomy-informed rubric is a good starting point) and picks the better one. That single judgment becomes one row of Dataset D: (clip, question, response_A, response_B, winner). Collect this across many clips and candidate pairs, and you have a training set directly usable for Bradley-Terry reward modeling (Section 10) or DPO (Section 8) with no further processing needed.

**Feedback type comparison**, cheapest to most expensive:

| Type | Example | Cost | Bradley-Terry compatible? |
|---|---|---|---|
| Binary success/failure | "Did it name the right hand? yes/no" | Very cheap | Not directly, but trivial as a rule-based reward |
| Pairwise preference | "Which explanation is better?" | Cheap, fastest to scale | **Yes, this is the native input format** |
| Ranking (3+) | Full ranking of several candidates | Moderate | Decomposable into multiple pairwise comparisons |
| Scalar ratings | "Rate 1-5" | Moderate | Only indirectly, prone to scale drift across sessions |
| Critiques | "This is wrong because..." | Expensive | No, needs further processing |
| Corrections | Human rewrites the flawed part | Expensive | No - used with dense/segment-level DPO variants (RLHF-V's approach, Section 3) |
| Expert demonstrations | Gold explanation written from scratch | Most expensive | No - used for SFT, not preference learning |

**RLHF, formally:** the umbrella strategy of the InstructGPT recipe (Ouyang et al., 2022, arXiv:2203.02155) - collect human preferences, fit a reward model to them (via Bradley-Terry, Section 10), then optimize the policy against that reward model with PPO or GRPO.

**RLAIF, formally:** identical structure, with an AI model doing the comparing instead of a human (Bai et al., 2022, "Constitutional AI," arXiv:2212.08073; Lee et al., 2023, arXiv:2309.00267). For your project specifically, this would mean using a larger/different VLM to judge which of two magic-trick explanations is better, instead of you doing it by hand. **This is worth proposing to Professor Xu as a scaling strategy for Stage 2 of the project**, once your own hand-labeled judgments (maybe 100-200) exist as a "gold" set you can use to check whether an AI judge agrees with you before trusting it to label thousands more automatically.

**Which did Professor Xu likely mean?** Given his explicit mention of "human demonstrations, labels, rankings, critiques, or preferences" as a single umbrella list in the confirmed pipeline, I read this as: he wants you to survey the *options* (which this section does) rather than having a single specific format already in mind. That reading is an inference, not a confirmed fact - it's worth asking directly (Section 17).

---

## 10. Bradley-Terry

**Level 1:** a simple rule for turning "A beat B" votes into numeric scores, repurposed in ML to turn "explanation A was preferred to explanation B" into a trainable reward number.

**Level 2 (formal):** Bradley & Terry (1952), *Biometrika* 39(3/4):324-345. Each item i has a latent strength p_i > 0; probability i beats j is P(i>j) = p_i/(p_i+p_j). Reparameterizing p_i = exp(r_i) gives P(i>j) = sigmoid(r_i - r_j). Training a reward model on preference data D = {(x, y_w, y_l)} minimizes:

```
L(theta) = -E[ log( sigmoid( r_theta(x, y_w) - r_theta(x, y_l) ) ) ]
```

**Level 3, numeric example (magic-trick-specific):** suppose your reward model currently scores r(explanation naming the false-transfer sleight) = 1.8 and r(explanation naming only "misdirection happened," no specific mechanism) = 0.6, for a clip where a human judge (you) preferred the more specific explanation.

- Difference: 1.8 - 0.6 = 1.2
- Predicted probability the model already assigns to the human's actual choice: sigmoid(1.2) = 1/(1+e^-1.2) ≈ **0.769**
- Loss for this example: -log(0.769) ≈ **0.263**

If the reward model instead had the scores reversed (r(specific) = 0.6, r(vague) = 1.8, difference = -1.2): predicted probability sigmoid(-1.2) ≈ **0.231**, loss ≈ **1.466** - a much larger penalty, correctly pushing the model to fix its disagreement with your judgment.

**Is Bradley-Terry necessary, or might Professor Xu mean something related?** As covered before, real alternatives exist (generative pairwise preference models that skip the separate scalar-score step; regression/hinge-loss variants like IPO or SLiC-HF, motivated by known Bradley-Terry limitations such as the transitivity assumption not always holding for real human judgments). **My assessment is unchanged: if feedback is collected as pairwise comparisons (the cheapest, most likely format given Section 9), Bradley-Terry is very likely exactly the right framework**, and I would not spend prototype time hunting for an exotic alternative unless Professor Xu specifically flags a reason to (e.g., if he's more interested in a *ranking* or *scoring* formulation tied to his own reward-machine/formal-methods background - see Section 18 - which would be a legitimately different, and worth directly asking about).

---

## 11. Reward function options

Your six requested formulations, each defined as precisely as the task allows:

**Reward 1 - Answer correctness.** `reward = 1 if model_answer == ground_truth_answer else 0` (or partial credit for near-misses on multi-part answers). **Data source:** Dataset B/D's objective ground truth. **Learned or programmatic:** programmatic. **Reward hacking risk:** the model could learn to always output the statistically most common answer across the dataset rather than reasoning per-clip, if answers aren't balanced. **Validation:** check performance on a held-out set with a *different* answer distribution than training.

**Reward 2 - Hidden-state correctness.** A specific case of Reward 1, scoped to "where is the hidden object/card now." **Data source:** Dataset B. **Programmatic.** **Reward hacking risk and validation:** identical to Reward 1, with the added risk that camera framing accidentally leaking the answer would make the reward trivially satisfiable without real reasoning - audit your footage for this specifically.

**Reward 3 - Causal correctness.** `reward = 1 if identified_action_range overlaps with true_causal_action_range else 0`, optionally with partial credit via temporal IoU (intersection-over-union of the predicted vs. true time window), matching how Time-R1 (Section 3, referenced via STVG-R1's citations) reports IoU-based rewards outperforming plain token-level supervision for temporal grounding. **Data source:** Dataset C, requiring you to annotate the true causal moment per clip. **Programmatic**, though annotating the "true" causal moment itself may require a subjective judgment call by you, which is a soft dependency on human labeling even though the reward computation itself is automatic. **Reward hacking risk:** the model could learn to point at whatever is most visually salient (fast motion) rather than what is causally responsible - this is exactly the failure mode Video-R1's temporal-shuffle contrast (Section 3/6) was built to catch, and adapting that mechanism here is a concrete, well-precedented mitigation. **Validation:** compare performance on correctly-ordered vs. temporally-shuffled versions of the same clip; a model doing genuine causal reasoning should perform much better on the ordered version.

**Reward 4 - Explanation quality.** A learned Bradley-Terry reward model (Section 10) trained on Dataset D's pairwise judgments, applied to score new candidate explanations at inference/training time. **Data source:** Dataset D. **Learned.** **Reward hacking risk:** length bias and confidence-bias (models learning to sound more authoritative rather than be more correct) are well-documented failure modes in the general reward-modeling literature (e.g., arXiv:2507.07375 explicitly names response length and style as common spurious correlations). **Validation:** periodically have a human directly grade post-training outputs against actual correctness, not just against the reward model's opinion, to check the two haven't co-drifted into agreeing with each other while both being wrong.

**Reward 5 - Evidence consistency.** `reward = alignment_score(explanation_text, visual_evidence)`, e.g., using the model's own attention/grounding behavior, or a simpler proxy like checking whether specific claimed objects/hands mentioned in the explanation are actually detectable in the corresponding video region at the claimed time. **Data source:** could be programmatic (an auxiliary detector) or partially human-graded. **Reward hacking risk:** a model could learn to describe generic, always-true visual facts ("a hand moved") to trivially satisfy a loose consistency check without engaging with the actual trick mechanism. **Validation:** spot-check high-reward explanations for exactly this kind of vacuous-but-technically-consistent pattern.

**Reward 6 - Hybrid reward.** `reward = w1 * causal_correctness + w2 * explanation_preference_score`, combining Reward 3 (or 1/2) with Reward 4. **Data source:** both Dataset B/C and Dataset D. **Hybrid (programmatic + learned).** **Reward hacking risk:** combining signals with different scales/noise properties without careful normalization is a well-known general RLHF pitfall; also risks "keyword stuffing" (inserting the technically-correct term into an otherwise weak explanation purely to satisfy the programmatic half). **Validation:** manually inspect high-reward outputs specifically for keyword-stuffing; adjust weighting if found.

**My recommendation for a first experiment:** **Reward 3 (causal correctness with a temporal-shuffle validation check) for Task B/Dataset C, and Reward 4 (Bradley-Terry explanation-quality) for Task A/Dataset D.** These are the two that most directly match Professor Xu's likely priorities (Section 4) while remaining implementable with the smallest dataset (Section 6).

**A note connecting to Professor Xu's own prior work, flagged clearly as inference, not fact:** his highly-cited 2020 paper "Joint inference of reward machines and policies for reinforcement learning" (confirmed via his Google Scholar profile, Section 18) is about **reward machines** - a formal, finite-automaton-based way of specifying reward functions for tasks with structured, non-Markovian temporal logic (e.g., "reward only after event A happens, then B, but not if C happens first"). This is a strikingly good conceptual fit for a causal-magic-trick reward (the reward genuinely does depend on a specific temporal/causal *sequence* of events, not just a single frame), and it is plausible he may want you to think about reward specification in these more formal, structured terms rather than a purely scalar Bradley-Terry framing. **I want to be explicit that I have no evidence he has said this about your project specifically - this is my own pattern-match to his publication record, not something he told you, and it belongs in the "strong inference" category of Section 18, worth raising as a direct question rather than assuming.**

---

## 12. Recommended training pipeline

**Method comparison, updated with the two-task structure from Section 5:**

| Approach | Data needs | Compute | Implementation complexity | Stability | Multimodal-compatible tooling confirmed? | Research value alone |
|---|---|---|---|---|---|---|
| SFT (supervised fine-tuning) | Gold explanations or answers | Low | Low | High | Yes, standard | Low - establishes a baseline, not itself a contribution |
| DPO | Preference pairs (Dataset D) | Low (no sampling loop) | Low-medium | High | Yes - TRL's `DPOTrainer`, confirmed VLM-compatible per current docs | Medium |
| GRPO | Rule-based reward (Dataset B/C) or learned reward | Medium (needs multi-sample generation, ideally vLLM) | Medium | Generally more stable than PPO for reasoning-style tasks, per the DeepSeek results and Video-R1's reported gains | Yes - TRL's official `grpo_vlm.py`, verl, OpenRLHF all confirmed | High, especially combined with the temporal-shuffle validation |
| PPO | Reward model + more infrastructure | High | High | Can be finicky (clip range, KL coefficient tuning) | Yes but heaviest to set up | Low marginal value over GRPO for this specific task shape, per current literature |
| Hybrid (SFT -> DPO -> GRPO staged) | All of the above, staged | Staged, starts low | Staged | Each stage validates the next | Yes | **Highest** - lets you report which specific stage contributed what, which is itself a finding |

**Staged recommendation, confirmed against the literature rather than assumed correct:**

1. **Baseline evaluation** (no training) - establishes your starting point (matches Video-R1 and VideoChat-R1's own practice of reporting zero-shot baselines before any RL).
2. **Small SFT pass** (optional, only if you have time to write a handful of gold examples) - a stronger baseline to compare later stages against, and a check on whether "just imitate good examples" already captures most of the improvement.
3. **DPO on Dataset D/Task A** - cheapest way to validate the full data-to-training loop.
4. **GRPO on Dataset B or C/Task B** - the RL stage, using a rule-based reward, extended with a temporal-shuffle validation check modeled on Video-R1's T-GRPO.
5. **PPO** - only if GRPO's results specifically suggest a need for a learned critic (e.g., if group-relative normalization proves too noisy at your small batch sizes) - not a default next step.

This progression matches your own suspected ordering (baseline -> SFT -> preference data -> DPO -> GRPO/PPO only if justified) and I found nothing in the current literature that contradicts it for a task of this shape and scale - if anything, Video-R1 and VideoChat-R1 both essentially validate exactly this "start with rule-based GRPO on a well-defined sub-task before anything fancier" approach for video-temporal reasoning specifically.

**Software stack, verified against current documentation:**
- **PyTorch + Hugging Face Transformers**: base layer; Qwen-VL models confirmed compatible via `AutoModelForImageTextToText`/`AutoProcessor`.
- **TRL**: confirmed official GRPO-for-VLM support (`grpo_vlm.py`, and the "Vision Language Model Alignment in TRL" blog post), plus `DPOTrainer` for the earlier stage. **Recommended as your primary framework** - it is the most beginner-friendly of the three RL frameworks checked, specifically because it already has a worked VLM example.
- **PEFT/LoRA**: confirmed compatible and recommended by TRL's own VLM docs; also directly supported by the `2U1/Qwen-VL-Series-Finetune` repo, which additionally supports video training, DPO, and GRPO together for the Qwen-VL family specifically.
- **vLLM**: confirmed integrated into TRL's GRPO trainer for fast multi-sample generation, important since GRPO needs several completions per prompt at every step.
- **verl / OpenRLHF**: both confirmed to support vision-language/multimodal RL (verl explicitly names Qwen2.5-VL support; OpenRLHF added VLM RLHF support, including multi-turn image-in-the-loop training, per its documentation) - more production/scale-oriented than TRL, worth learning as a second stack once you outgrow TRL's simplicity, not as your starting point.
- **Not recommended:** classic robotics-RL libraries (e.g., stable-baselines3) - I found no evidence they are used anywhere in the current VLM-RLHF literature; they are built for continuous-control environments, a genuinely different RL problem shape than token-generation policies, despite the surface-level similarity to your existing RL-adjacent robotics background.

---

## 13. Evaluation methodology

**Held-out set, non-negotiable from day one:** carve off a fixed 20-25% of clips before any training or iterative prompt-tuning, split by *trick and performer*, not just by clip, so the same physical setup never appears in both halves.

**Core metrics, mapped to your two tasks:**
- Task B (hidden-state): exact-match accuracy, broken out per trick type.
- Task A (explanation): blind pairwise win-rate of post-trained vs. pre-trained outputs, plus rubric-based scoring along the misdirection-taxonomy dimensions (did it identify the attentional misdirection? the physical mechanism? both, separately scored, so a low score is diagnosable rather than a single vague number).

**Your central methodological question, restated and answered directly: how do we tell memorization apart from genuine reasoning improvement?** Given your realistic scale (dozens, not thousands, of clips), the most defensible tools are the generalization axes you listed, each isolating a different failure mode:
- **Unseen tricks:** does improvement hold on trick types never seen during training/preference collection? This is the single most important check - if improvement only shows up on training-resembling clips, that's a strong memorization signal.
- **Unseen performers:** does it transfer across different hands/performance styles? Tests whether the model learned something about *tricks* rather than something narrow about your specific footage.
- **Unseen camera positions:** brittleness to camera angle is a red flag for shallow/superficial learning.
- **Unseen props:** does a coin-trick-trained model's improvement transfer to a card trick using a structurally similar mechanism (e.g., misdirection via a verbal distraction), or is it fully prop-specific?
- **Unseen wording:** rephrasing the same question should not change the answer if the model is reasoning about the video rather than pattern-matching the question's exact phrasing.
- **Unseen variations of a known trick:** the middle ground between "identical to training" and "fully novel trick" - useful for measuring how far generalization extends along a continuum rather than as a binary.

**A second, quantitative check specific to your reward-hacking concern:** compare the *reward function's own* score improvements (from Section 11) against *independently held-out, human-graded* improvements. If the reward-model or rule-based score goes up substantially more than actual graded quality, that's a direct, numeric signal of reward hacking rather than genuine task improvement - this doesn't fully solve the problem (a genuinely open methodological challenge across the whole field, not unique to your project), but it gives you an honest, reportable basis for your claims either way.

**Temporal-shuffle check (borrowed directly from Video-R1's own validation practice, Section 3):** if your model's performance on Task C-style causal questions is similar whether the input video is correctly ordered or randomly shuffled, that's strong evidence it isn't doing genuine temporal reasoning at all - it's a clean, cheap, and already-precedented diagnostic worth adopting directly rather than inventing your own from scratch.

---

## 14. Novelty assessment

**What has already been done:** the RLHF/RLAIF/DPO/GRPO methodology for VLMs generally (LLaVA-RLHF, RLHF-V, RLAIF-V, VLM-R1, Video-R1, VideoChat-R1) is well established. Reusing this methodology unchanged on a new domain is not, by itself, a research contribution.

**What is incremental:** applying an existing recipe (e.g., TRL's GRPO-for-VLM path, or Video-R1's T-GRPO reward) to magic-trick video with no methodological changes. Necessary as a learning phase (Section 15), not sufficient as a contribution on its own.

**Ranked possible contributions:**

1. **A magic/mentalism visual-reasoning benchmark, framed in CLEVRER's descriptive/explanatory/predictive/counterfactual taxonomy.** *Novelty:* high - no such benchmark found anywhere in my searches. *Feasibility:* high, given a filming setup and a few dozen tricks. *Compute:* low. *Data:* moderate, achievable solo. *Experimental clarity:* high if the task is well-posed. *Research value:* high - even "here's a benchmark and baseline scores across several open VLMs" is a complete, legitimate contribution on its own.
2. **Extending Video-R1's temporal-shuffle-contrast reward mechanism to a domain where temporal order is adversarially exploited (misdirection) rather than incidentally important.** *Novelty:* high - this is a genuinely different stress test than the general video content T-GRPO was built and tested on. *Feasibility:* medium (requires Dataset C plus a working GRPO pipeline first). *Compute:* medium. *Experimental clarity:* medium-high, given the built-in shuffle-based diagnostic. *Research value:* medium-high, and connects most directly to Professor Xu's own published research focus (causal/temporal RL - Section 18).
3. **A domain-specific reward-hacking case study** (does a small preference-trained reward model get gamed in magic-trick-specific ways, e.g., confidently naming a physically plausible but wrong mechanism because it "sounds like" a correct answer?). *Novelty:* medium-high. *Feasibility:* high - this largely falls out of doing Sections 11-13 carefully and reporting honestly. *Research value:* high, and a genuinely useful negative/messy result is still a real finding here.
4. **A full causal-reasoning study comparing pre- and post-training performance specifically on counterfactual questions** ("what if the magician hadn't done X"), directly building on CausalVLBench/CounterVQA's finding that even general-purpose VLMs struggle badly on multi-hop counterfactual chains. *Novelty:* high. *Feasibility:* lower - counterfactual ground truth for a real (non-synthetic, non-simulated) magic trick is genuinely hard to establish, since you can't "re-run" a physical trick with a controlled intervention the way CLEVRER's synthetic physics engine can. *Research value:* high if feasible, but I'd flag this as a stretch goal, not a first-prototype target.
5. **Physical trick performance/reproduction (interpretations F/G from the earlier document).** Explicitly ruled out by Professor Xu's clarification (Section 1) - not recommended at any stage of this specific project, though possibly a legitimate longer-term direction if the lab's robotics infrastructure gets involved later.

**Honest bottom line:** the strongest, most achievable combination is **#1 (benchmark) + #3 (reward-hacking case study)**, with **#2 (temporal-shuffle extension)** as a natural, well-precedented next step that would most directly connect your project to Professor Xu's own research program if that connection turns out to be real (confirm at your next meeting, Section 17).

---

## 15. Minimal technical prototype

Your proposed eight-stage progression, checked against the literature and lightly adjusted:

| Stage | Objective | Approx. dataset size | Compute | Implementation difficulty | Expected result | Scientific value | What could go wrong |
|---|---|---|---|---|---|---|---|
| 0 | Choose one VLM | N/A | None | Low | A concrete, justified choice (Section 7: Qwen2.5-VL-7B recommended) | Low on its own, but unblocks everything else | Choosing based on hype rather than the concrete tooling/precedent evidence in Section 7 |
| 1 | Tiny magic-trick benchmark | 15-25 clips | Filming time, no GPU needed yet | Low-medium (schema design matters more than volume) | A working Dataset B (Section 6) with objective ground truth | High - this is Professor Xu's explicitly requested first deliverable | Camera framing accidentally leaking or fully hiding the answer (Section 6) |
| 2 | Evaluate the unmodified VLM | Stage 1's clips | Minutes, small GPU | Low | Baseline accuracy/quality numbers, likely mediocre | Establishes your starting point; also a sanity check that the task is well-posed (not already trivial, not literally unanswerable) | Base model may refuse, hedge, or misunderstand the prompt format before you've even gotten to training |
| 3 | Collect human preference labels | 2-4 candidates x each clip, ~60-100 pairwise judgments | Your time, no GPU | Low technically, moderate time cost | Dataset D | Tests whether your rubric (Section 9) is usable and consistent | Annotator (your own) inconsistency across sessions |
| 4 | Train a preference/reward model, if justified | Stage 3's data | Single GPU, likely under an hour | Low-medium | A Bradley-Terry reward model with some, likely imperfect, held-out agreement | Tests whether reward modeling works at all on a dataset this small - a genuine, reportable finding either way | Overfitting to ~100 pairs is a real risk worth reporting honestly rather than hiding |
| 5 | Supervised fine-tuning (optional) | A handful of hand-written gold explanations, if time allows | Single GPU, short run | Low | A stronger imitation baseline | Tests whether "just imitate good examples" already captures most of the gain | May not be worth the time investment if Stage 3-4 are already consuming most of your week |
| 6 | DPO | Stage 3's preference pairs | Single GPU, no sampling loop | Medium | Measurable shift toward preferred-style outputs | Cheapest way to test whether preference training changes behavior in the intended direction at all | "Likelihood displacement" (arXiv:2410.08847) - model avoids the rejected wording without improving the underlying reasoning |
| 7 | GRPO, with a rule-based and/or temporal-shuffle reward | Stage 1's clips as live prompts | Single GPU with vLLM, longer run | Medium-high | A model scoring measurably higher on your reward function, *if* the reward is well-designed | Tests whether online RL post-training genuinely improves the target reasoning, using the shuffle-contrast diagnostic from Section 13 | Reward hacking, training instability, running out of time before convergence - all realistic and all worth reporting honestly |
| 8 | Compare all methods on a genuinely held-out set | Held-out portion, never touched before | Minimal additional compute | Medium (evaluation design is the hard part) | A clear before/after/across-methods comparison table | The actual scientific payoff of the whole project | Train/test contamination if the held-out set wasn't fixed and untouched from the very start |

**Translating Professor Xu's three-step instruction into your first 3-5 concrete deliverables (Part 18 of your request):**
1. **The literature table from Section 3** (already produced here) - Professor Xu's instruction #1, "survey relevant VLM post-training methods," done.
2. **A written dataset schema for Task B and Task A** (Section 6, Datasets B and D specifically), with exact input/output/ground-truth format specified.
3. **10-20 real filmed trick moments**, following that schema - the actual "small prototype dataset" he asked you to think about, made concrete rather than theoretical.
4. **A baseline evaluation script and results** (Stage 2 above) - evidence you've actually run a real VLM against real footage, not just planned to.
5. **A short written note on the two-task framing (Task A/B from Section 5) with your reasoning**, so the professor can correct your scoping before you sink more time into the wrong one.

---

## 16. 7-day execution plan

**Day 1:** Set up Python/PyTorch/Transformers/TRL/PEFT locally or on a rented GPU. Download Qwen2.5-VL-7B-Instruct (or 3B if compute-constrained) and confirm basic image+text inference works.

**Day 2:** Film or source 10-15 short trick clips for Dataset B (hidden-state reasoning) - prioritize tricks where you can verify the actual mechanism, and vary hiding location/prop across clips deliberately (Section 6's bias warning).

**Day 3:** Write the exact input/output schema for Task B (question templates, answer format). Run Stage 2 (baseline evaluation) on all clips; record where the model succeeds/fails, honestly.

**Day 4:** Build Dataset D from the same clips: sample 2-4 candidate free-text explanations per clip at temperature ~0.8, begin pairwise preference judgments using a rubric derived from Section 4's misdirection-taxonomy framing.

**Day 5:** Finish preference collection (target 60-100+ pairs). Work through TRL's official GRPO-VLM documentation with a toy (non-project) example to get comfortable with the API before touching your real data.

**Day 6:** Attempt a first real DPO run on Dataset D. If time and stability allow, attempt a first GRPO run on Dataset B using the exact-match reward from Section 11 (Reward 1/2). Document what worked and what broke, honestly, including partial/failed runs.

**Day 7:** Consolidate: baseline results, whatever training results you obtained, and a short written summary using this document's Section 17 questions as your talking points. Bring this document as reference, not as something to present verbatim.

---

## 17. Questions for Professor Xu

### What I should be able to explain to Professor Xu
Before the meeting, you should genuinely (not superficially) understand: what a VLM is and how it differs from a text-only LLM; the difference between pretraining, SFT, and post-training (Section 2); what a reward function is and the difference between a programmatic and a learned one (Section 11); what a preference/reward model is and how Bradley-Terry turns pairwise judgments into one (Section 10); how DPO works and why it needs no reward model (Section 8); how PPO and GRPO differ, specifically why GRPO drops the critic network and what "group-relative" actually means numerically (Section 8's worked example); what RLHF and RLAIF are and how they differ only in who provides the preference labels (Section 9); why a benchmark needs held-out generalization axes to distinguish memorization from real reasoning, not just a single accuracy number (Section 13). You do not need to sound like an expert - you need to be able to explain each of these in your own words, including where you're still uncertain.

### Substantive questions to ask
1. Should the project prioritize **hidden-state/causal reasoning** (Task B, Section 5) or **free-text mechanism explanation** (Task A), or does he want both pursued in parallel as this document recommends?
2. Does the project connect to his own published work on **temporal causal reinforcement learning** and **reward machines** (Section 18) - and if so, should the reward function be a formal/structured specification rather than a plain scalar Bradley-Terry-style reward?
3. Does he expect **human-only feedback**, or is an **RLAIF-style AI-judge extension** (Section 9) an acceptable/preferred way to scale past a small hand-labeled set?
4. Is there an existing compute budget, cloud GPU access, or a specific base VLM already used in his lab that should override the Qwen2.5-VL-7B recommendation in Section 7?
5. Is self-filmed footage the right sourcing strategy, or does he have access to existing trick footage/collaborators (e.g., a magician consultant) that would change the dataset design in Section 6?
6. How ambitious should the counterfactual-reasoning capability (#8, Section 4) be for a first version - full multi-hop counterfactual questions (per CounterVQA's finding that even general VLMs struggle here), or should it be deferred to a later stage as this document recommends?
7. Does he want the project framed toward a specific venue or outcome (a lab report, a workshop paper, a longer-term thesis-style project), which would change how much of Section 14's ranked novelty list is realistic to target?
8. Is there a reason to prefer PPO over GRPO for this specific task that isn't apparent from the current literature (Section 8), given his own RL background is broader than the LLM-specific GRPO/PPO literature I searched?

---

## 18. Confirmed / inferred / unknown ledger

### Confirmed by Professor Xu
- The eight-stage pipeline you proposed (base VLM -> dataset -> prompts -> human feedback -> reward/preference model -> post-training via PPO/GRPO/DPO/related -> optimization -> evaluation) is broadly correct.
- Magic and mentalism are the literal, concrete domain - not a metaphor for something else.
- The primary purpose is visual reasoning, inference, and explanation from demonstrations - not physical trick performance.
- The immediate next steps are: survey VLM post-training methods, design a small prototype dataset, narrow scope based on feasibility.
- Evaluation, specifically distinguishing genuine reasoning improvement from superficial metric gains, is explicitly important to him.

### Strong inference
- Professor Xu's own public research record (ASU faculty page, Google Scholar profile, ASU News feature, and his lab's own website, all checked directly today) shows his primary research areas are **cyber-physical systems, control theory, reinforcement learning, formal methods, and robotics**, with an active NSF CAREER award specifically titled **"Temporal Causal Reinforcement Learning and Control for Autonomous and Swarm Cyber-Physical Systems"** (2024-2029, ~$550,000). His lab is named the **Neuro-symbolic Intelligent Cyber-physical systEms (NICE) Lab**, and its own recruiting page states interest in "learning and control of cyber-physical systems using frontier large language models (e.g., LLM, VLM, VLA, MLLM)." He has a highly-cited prior paper on **reward machines** (a formal, automaton-based reward specification method) for reinforcement learning. **None of this was stated to you directly about the magic-trick project - it is my own pattern-match between his public research record and your project, and should be treated as a hypothesis to raise with him, not as a settled fact about his intentions.** I've corrected the earlier document's speculative connection to ASU's cognitive-science-of-misdirection literature (Kuhn/Goldinger) - given what his actual research record shows, that connection now looks like a coincidence rather than a real link, and I'd deprioritize it.
- Video content, not just still images, is very likely necessary for most of the plausible task interpretations (Section 4/7).
- Task A (explanation) and Task B (hidden-state) are the two most technically tractable and most likely-intended interpretations, given the confirmed pipeline's emphasis on reward/preference-model-judgeable outputs.

### Still unknown
- Whether Task A, Task B, both, or a different interpretation entirely is the actual intended scope.
- Whether human feedback is expected to be exclusively human-provided, or whether an AI-judge/RLAIF extension is acceptable.
- Whether Professor Xu has an existing dataset, compute budget, specific VLM, or specific reward-formalism (e.g., reward machines) already in mind.
- Whether video sourcing should be self-filmed or whether there's an approved existing source/collaborator.
- Whether the counterfactual-reasoning capability should be attempted in a first pass or deferred.
- Whether this connects, in his mind, to his broader causal/temporal-RL research program, or is a more self-contained exploratory project.

---

## Proposed project specification

### Working title
*Visual Reasoning and Explanation from Magic and Mentalism Demonstrations: A Post-Training Study of Vision-Language Models*

### Research question
Can preference- and/or reward-based post-training (DPO and/or GRPO) improve an open vision-language model's ability to correctly infer hidden state and explain the causal mechanism behind magic/mentalism demonstrations, in a way that generalizes to unseen tricks, performers, and camera angles rather than merely fitting the small training set?

### Input
A short (10-30 second) video clip of a single magic or mentalism demonstration, plus a natural-language question (for hidden-state reasoning) or instruction (for explanation).

### Output
Either a specific hidden-state answer with justification (Task B), or a free-text causal explanation of the trick's mechanism (Task A).

### Dataset
A self-filmed set of 15-25 trick clips (Dataset B/Section 6), reused to generate a pairwise-preference set of 60-100+ judgments over model-generated explanations (Dataset D/Section 6), split by trick and performer into training and a fixed, untouched held-out evaluation portion from day one.

### Baseline model
Qwen2.5-VL-7B-Instruct (or the 3B variant if compute-constrained), with LoRA adapters - chosen because it is the base model independently used by both of the closest existing methodological precedents (Video-R1 and VideoChat-R1), and because it has the most directly confirmed GRPO/DPO/video-training tooling available today (Section 7/12).

### Human feedback
Pairwise preference judgments over model-generated explanations, collected by hand against a rubric derived from the psychology-of-magic misdirection taxonomy, with an optional RLAIF-style AI-judge extension once a small human-graded "gold" set exists to validate it against.

### Preference/reward model
A rule-based exact-match reward for Task B (Reward 1/2, Section 11), and a small Bradley-Terry preference reward model trained on the pairwise data for Task A (Reward 4, Section 11), with an explicit reward-hacking validation step comparing reward-model-scored improvement against independently held-out, expert- or human-graded improvement.

### Post-training method
DPO first, on Task A's preference data, to validate the pipeline cheaply; GRPO second, on Task B's rule-based reward (optionally extended with a Video-R1-style temporal-shuffle contrast reward for causal-moment localization); PPO deferred unless GRPO's results specifically indicate a need for it.

### Evaluation
Exact-match accuracy for Task B and blind pairwise win-rate plus taxonomy-based rubric scoring for Task A, both measured across five generalization axes (unseen tricks, performers, camera angles, props, and question wording), plus a temporal-shuffle diagnostic to directly test whether any improvement reflects genuine temporal/causal reasoning rather than superficial pattern matching.

### First experiment
Stage 1-2 from Section 15: build the 15-25 clip Dataset B, and run a zero-shot baseline evaluation of Qwen2.5-VL-7B-Instruct on it, documented honestly, as the concrete artifact to bring to your next meeting alongside this document's open questions (Section 17).

### Expected contribution
At minimum, a small novel benchmark for magic/mentalism visual reasoning with documented baseline scores - a complete, achievable, honest contribution on its own, given that no such benchmark currently exists in the literature I could find. Potentially, if time and results allow: an empirical finding on whether post-training genuinely improves causal/temporal visual reasoning in a domain engineered to contain adversarial misdirection, and/or a documented, domain-specific reward-hacking case study.
