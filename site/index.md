---
layout: default
title: Xyntetik
permalink: /
description: Local models on hardware you own. Runner is the engine, free forever under Apache 2.0.
---

# Xyntetik

Local models on hardware you own. Xyntetik is independent and
bootstrapped: **the engine is free forever under Apache 2.0**, and
consulting and enterprise work fund the hardware. Built in Sweden. Your
data never leaves the building.

## Runner

A single-binary GGUF inference and LoRA training engine written from
scratch in plain C, for CPU, CUDA and Metal. It trains LoRA adapters
through the same quantized weights it serves, closes tool calls that run
out of tokens so they still parse, and records every run as a receipt that
replays. Every claim on these pages is tied to a measurement you can re-run.

- [Runner](/runner/): what it is, sixty seconds to a served model, why this and not llama.cpp
- [Train a LoRA on the quantized GGUF you serve](/runner/train-lora-on-quantized-gguf/)
- [Reproducible LoRA training with receipts](/runner/reproducible-lora-training-receipts/)
- [Tool calls that survive the token limit](/runner/truncation-safe-tool-calling/)
- [Benchmarks](/runner/benchmarks/), [models and published artifacts](/runner/models/), [support matrix](/runner/support-matrix/), [all docs](/runner/docs/)
- [Source, releases and issues on GitHub](https://github.com/Joakimpalm-Zen/xyntetik-runner)

## Research

Measured reports and artifacts are published on
[Hugging Face](https://huggingface.co/Joakimpalm-Zen): quantization fidelity
ladders, pruned and surgically reduced models with their envelopes, and the
adapters the training claims are measured on.
