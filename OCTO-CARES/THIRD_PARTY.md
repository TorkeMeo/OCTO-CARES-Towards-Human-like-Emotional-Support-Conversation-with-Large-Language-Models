# Third-Party Sources and Release Conditions

This file records third-party protocol sources visible in the included code.
It does not grant redistribution rights to upstream code, datasets, API output,
or model weights. No private datasets or checkpoints are bundled.

## MultiAgentESC

- Upstream repository: <https://github.com/MindIntLab-HFUT/MultiAgentESC>
- Recorded commit: `631b7f1961fc7502e547fd9258e847230dbcb973`
- Source adaptation: `experiments/generate_multiagentesc_official_no_rag_qwen3.py`
- ESCoT v2 wrapper: `experiments/generate_escot_seeker_only_reddit_multiagent_v2.py`

The adaptation preserves the documented complexity gate, sequential analysis,
strategy-agent discussion, strategy-conditioned candidates, debate, reflection,
majority selection, tie judge, and final refiner. It removes SBERT retrieval,
Top-10 ESConv examples, and all retrieved demonstrations. All roles use the
same supplied local Qwen3 checkpoint. The group-chat control flow is implemented
directly instead of requiring the upstream AutoGen runtime. The v2 wrapper adds
seeker-background/latest-turn separation and Reddit-style final replies.

This baseline must be described as a no-retrieval project adaptation, not as a
complete reproduction of the original system. Upstream provenance constants
are retained in generated metadata.

## EmotionalRAG

- Upstream repository: <https://github.com/BAI-LAB/EmotionalRAG>
- Recorded commit: `2f932e027a6edc247a222b50353490bf557623b5`
- Retrieval source: <https://github.com/BAI-LAB/EmotionalRAG/blob/2f932e027a6edc247a222b50353490bf557623b5/get_response.py>
- Adaptation: `representation/emotional_rag_retrieval.py`
- Emotion preparation: `representation/prepare_emotional_rag_reddit1053.py`

The adaptation records the five retrieval rules and the upstream eight emotion
axes/intensity scale. It changes the task to leave-one-post-out Reddit
retrieval, uses explicit same-ID exclusion and stable corpus-order tie
breaking, and evaluates retrieved posts with subreddit and label-overlap
metrics. It does not reproduce upstream role-play response generation.

Encoders are explicit inputs. The Qwen base-cache setting is a resource-aware
semantic-encoder substitution and must not be described as the same BGE-based
experimental setup. Emotion estimates are also produced by the supplied local
model, rather than being assumed identical to upstream estimates.

## Data and Model Dependencies

The experiments refer to Reddit posts/comments, ESCoT contexts, ESConv strategy
definitions, eight-label API annotations, and externally obtained checkpoints.
Their access, privacy, platform terms, licenses, and redistribution permissions
must be checked independently. The data schema documentation is not a dataset
license. Remove usernames, post URLs, IDs that expose identities, and private
content from public examples unless their release is specifically justified
and permitted.

Recorded model families include Qwen3-8B, qwen3.7-max API annotation,
gpt-oss-120b summary generation, and local GLM-4-32B, Gemma-3-27B, and
Qwen3.5-27B judges. Model weights and provider credentials are not distributed.
Follow each checkpoint's or provider's applicable conditions, and record the
actual model revision in new experiments. Downloading a model is not implicit
permission to redistribute it or its inputs.

Runtime dependencies include PyTorch, Transformers, PEFT, NumPy, the OpenAI
Python client, and optional FlagEmbedding. Local serving uses external vLLM
Docker images and an NVIDIA-compatible container runtime. Their respective
licenses remain applicable; a dependency list is not a replacement for
third-party notices.

## License Verification Pending

The license texts and redistribution terms at the two pinned upstream commits
have not been independently verified by this packaging step. A permissive
license must not be assumed from public GitHub availability. Before publishing
the package, review the pinned source trees, determine whether copied or
adapted prompt/code material requires notices or permission, and include all
required license texts without altering their attribution.

The package's own license should be chosen only after that review. Until the
review is complete, this document is a provenance inventory, not a declaration
that the entire package can be distributed under a single license. An anonymous
research submission must still retain legally required third-party attribution;
anonymity does not override license obligations.
