# 🐾 VLM_dog_gait

This repository contains the full implementation of my **Bachelor Graduation Project** at the University of Twente (Creative Technology BSc), titled:  
**“The Use of Vision-Language Models in Video-Based Canine Gait Assessment through Chatbots.”**

---

## 🧠 Project Summary

The goal of this project was to design a virtual veterinary assistant that analyzes video footage of dogs walking to detect signs of **lameness** and **musculoskeletal disorders**. The system is deployed as a **Telegram chatbot**, powered by a **LoRA fine-tuned Video-LLaMA3‑7B** model.

It enables real-time gait evaluation through a multi-turn prompt flow, compares fine-tuned and base model outputs, and provides veterinary-style diagnostic feedback.

---

## 🏗️ Structure

| Folder | Description |
|--------|-------------|
| `610v 4/` | **LoRA fine-tuning dataset**: 29 video samples annotated via `train.json`, used for adapting the base model to canine gait-specific patterns. |
| `train/` | Full training pipeline: `train_lora.py` handles dataset loading, video frame extraction, and LoRA training; `start_training.py` automates cluster-based training job submission via SLURM. |
| `Analysis Dog Bot - Base and Lora 0.3/` | Final system implementation: includes Telegram bot logic, model inference comparison scripts, and cluster job submission tools. Runs on user-submitted video and outputs structured diagnostic feedback. |

---

## 📸 Telegram Bot Demo

The chatbot allows users to submit a short video of their dog walking. It then asks for contextual info (age, breed, symptoms) and returns a model-generated analysis comparing **base** and **fine-tuned** model outputs.

Key scripts:
- `main.py` — bot logic, step-by-step interaction
- `tocluster.py` — handles HPC job submission and results
- `analysis.py` — dual inference: fine-tuned vs. base model

---

## 🧪 LoRA Fine-Tuning

- Model: `Video-LLaMA3‑7B` from DAMO-NLP-SG
- Optimized using HuggingFace + PEFT (LoRA)
- Dataset: 29 custom-labeled videos (`train.json`)
- 610×610 resolution, 10 seconds, 30 FPS
- Training done on an **NVIDIA L40 GPU (cluster)**

---

## ⚙️ Tech Stack

- 🐍 Python 3.10
- 🤗 HuggingFace Transformers & PEFT
- 🔬 Vision-Language Model: Video-LLaMA3‑7B
- 🎓 LoRA fine-tuning
- 💬 Telegram Bot API
- 🧠 Custom veterinary prompt design
- 🧵 SLURM cluster job management (paramiko + SSH/SFTP)

---

## 🔒 Git LFS Required

This repo uses [Git LFS](https://git-lfs.com/) for video data.  
To properly clone and access video files:

```bash
git lfs install
git clone https://github.com/sade-yakubova/VLM_dog_gait.git
cd VLM_dog_gait
git lfs pull

