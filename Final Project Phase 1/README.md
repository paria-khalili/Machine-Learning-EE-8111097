# Final Project, Phase 1 - Problem Formulation

Phase 1 of the final project, introducing and scoping a Spoken Language Identification (SLID) system: a model that classifies an input audio signal by language rather than transcribing its content.

## Contents

- `ML_ProjectDescription (1).pdf` — the official project brief/requirements handed out for the assignment.
- `phase1_report_810801065.pdf` — the Phase 1 report, covering:
  - Motivation for SLID and how it differs from ASR (speech-to-text)
  - Real-world use cases (voice-assistant language routing, emergency call-center routing, low-resource dialect archiving)
  - The core modeling goal: learning representations that are invariant to *how* or *where* the speech was recorded, so the model generalizes across speakers and recording conditions

## Relation to Phase 2

This phase sets up the problem and approach; the implementation (data cleaning, feature extraction, classification, clustering, and evaluation) is carried out in [Final Project Phase 2](../Final%20Project%20Phase%202).
