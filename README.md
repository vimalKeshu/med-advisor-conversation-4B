# Med Advisor Conversation 4B — Training Code

Training scripts and experiment configurations for **[Med Advisor Conversation 4B](https://huggingface.co/vmal/med-advisor-conversation-4B)**, a 4B-parameter medical education and conversational reasoning model built on Qwen3-4B.


## Training stages

| Stage | Purpose | Documentation |
| --- | --- | --- |
| **Supervised fine-tuning (SFT)** | Fine-tune the base model,**Qwen3-4B**, on medical conversations to establish conversational behavior. | [`sft-training/`](sft-training/) |
| **Reinforcement learning (RL)** | Post-train the SFT model to improve response quality and behavior. | [`rl-training/`](rl-training/) |


## Model release

**[vmal/med-advisor-conversation-4B on Hugging Face](https://huggingface.co/vmal/med-advisor-conversation-4B)**


## License

[Apache 2.0](LICENSE).