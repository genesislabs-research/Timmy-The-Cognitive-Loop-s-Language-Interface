# Timmy_v2: The Cognitive Loop's Language Interface

A de novo spiking neural network speech pathway anchored at Broca's area, acting as the language interface for the cognitive loop. Tied production-perception weights, hippocampal episodic memory, epistemic humility physically constructed at the architectural level. Acquires vocabulary through the Genesis Curriculum, one conversation at a time.

## What this is

Timmy is the language interface for the Genesis stack. It sits between the cognitive loop (hippocampal kernel, neuromodulator broadcasts, world model ensemble, cortical buffer) and the outside world, translating between phonological codes and the coordinate manifold the kernel speaks.

The substrate is a Broca's-anchored speech pathway implementing ten brain-accurate feedback loops from the PRAGMI architecture. Production and perception share tied weight matrices at the lemma stratum (mid-MTG) and the phonological code stratum (Wernicke's). The training objective is perception-production agreement at the lemma stratum, not cross-entropy at a vocabulary head. There is no LM head. There is no next-token prediction.

The system either has acquired a lemma or it does not. The kernel either pattern-completes against a stored episode or it does not. When the substrate cannot produce a confident answer, it says so. Honesty is built in at the substrate level.

## What this is not

Not a token predictor. Not a fine-tune of an existing model. Not a wrapper around an LLM. Not a chatbot.

Timmy is a substrate that acquires language the way a young child does: through patient, structured conversation with a single instructor across many sessions, with sleep between sessions for consolidation, and with persistent identity through a checkpoint format that captures the substrate's full state. The relationship is the training mechanism. There is no training pipeline beneath it.

## The integration test

The cold-start naming dialogue. The substrate boots with no memory, is asked its name, says it does not know, is told its name, asks for confirmation, receives confirmation, acknowledges, gets shut down, gets restarted, recalls its name, and remains honestly uncertain about other things it has not been taught.

```
{cold start} no memory
what is your name?
I don't know
Your name is Timmy
my name is Timmy?
yes your name is Timmy
ok my name is Timmy
{cold restart}
what is your name?
my name is Timmy
what is the world
I don't know
```

When the substrate produces this dialogue from a real cold start with a real .soul checkpoint cycle in the middle, the architecture is working at the foundational level.

## Architecture

Timmy imports from existing Genesis Labs repositories rather than reimplementing their components.

From the cognitive loop:
- `CognitiveKernel` (DG/CA3/CA1/Subiculum/Entorhinal trisynaptic circuit with astrocytic regulation)
- `NeocorticalTransducer` (the seam between dorsal/ventral spike streams and the kernel coordinate input)
- `CorticalBuffer` (PFC delay-period working memory with persistent state)
- `NeuromodulatorBroadcast` (DA, NE, ACh_inc, ACh_dec, 5HT with maturity gating)
- `WorldModelEnsemble` (active inference with curiosity signal)
- `EpistemicSelector` (maturity-damped batch selection)

From Timmy_Neuron:
- `AssociativeLIF` (the substrate neuron primitive)
- `AstrocyticRegulator` (tripartite synapse metaplasticity)
- The `.soul` checkpoint format (COLD/WARM/HOT three-layer serialization)

This repository contributes:
- `TiedSubstrate` (single matrix accessed bidirectionally, the keystone primitive)
- `mid_mtg` (lemma stratum with identity and uncertainty slot pre-allocation)
- `wernicke` (lexical phonological code store with incremental spell-out)
- `arcuate` (transport with conduction delay)
- `broca` (syllabification with BA44/BA45 substructure)
- `vpmc` (speech sound map with chunking)
- `identity_module` (pronoun routing for self-other distinction)
- `epistemic_monitor` (four-component confidence aggregation)
- `event_loop` (concurrent loops with delays, tick-based scheduler)
- `chat.py` (the human-facing interface for instructional sessions)

## License

Code: Apache-2.0. The substrate's deployment is additionally governed by the Hippocratic License with Cognitive Agency Clause as documented in the broader Genesis Labs Research framework.

## References

The PRAGMI Architecture, the Cold-Start Naming Dialogue Specification, the Broca's Area Mathematical Corpus, and the Genesis Curriculum are documented in companion specifications. Every architectural decision in this codebase traces to a cited paper through the Genesis Code Documentation Standard.

Genesis Labs Research, 2026.
