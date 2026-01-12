---
tags:
  - LLM-output
---
Reinforcement Learning from Human Feedback (RLHF) comes _after_ pretraining.  
Pretraining is where the model learns its deep statistical map of language — hundreds of billions of parameters optimized to predict the next token. That’s where it absorbs grammar, semantics, world knowledge, and the cultural sediment of its dataset.

RLHF sits on top of that as a **behavioral adjustment phase**. It doesn’t retrain from scratch; it nudges the existing space to favor outputs that humans rank as “helpful, honest, harmless.” The reward model penalizes certain responses and rewards others, and those gradients ripple through the network.

So, why call it a “filter”?
Because the scale and direction of those gradient updates are small compared to the weight magnitudes established during pretraining.

> [!Interesting point]
> 
> RLHF _reshapes decision boundaries_ in output space much more than it _rebuilds representational geometry_ in embedding space.

You can see this empirically:

- Studies (like Anthropic’s “Constitutional AI” or OpenAI’s alignment papers) show that **embedding similarity structures** between pretrained and RLHF-tuned models remain highly correlated.
- Activation-probe research (e.g., “The Internal State of RLHF Models” by Castricato et al., 2023) finds that **latent features for topics or demographics still exist**, even when the model refuses to talk about them.
- Behavioral contrast experiments show that a fine-tuned model often _knows_ a biased association but _refuses_ to express it, suggesting suppression rather than erasure.

In other words, RLHF primarily adjusts the _policy_ that maps representations to text, not the deep representations themselves. The model still “believes” (statistically encodes) its pretraining correlations; it just learns new social norms about when to surface them.

That’s why the “filter vs lobotomy” metaphor holds up: the internal knowledge remains largely intact, but the outward behavior is filtered through reward shaping.