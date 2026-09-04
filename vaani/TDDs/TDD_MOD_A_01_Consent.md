# TDD: Consent & Privacy System (MOD-A-01)
**Source Spec:** DOC6_CONSENT_PRIVACY.md
**Module:** A (Data Foundation)

## 1. System Overview
The Consent & Privacy system is the "legal firewall" of VAANI. It ensures that no audio is recorded, cloned, or published without explicit, itemized consent aligned with India's DPDP Act 2023. It transforms paper/PDF forms into a machine-readable register that governs all downstream data splits.

**Inputs:** Signed PDF/Paper forms (Form A, B, C).
**Outputs:** `consent_register.json` $\rightarrow$ Filter for `manifest.py`.

## 2. Component Architecture

### 2.1 The Consent Register (`consent_register.json`)
A central, encrypted JSON store mapping real identities to anonymous speaker IDs.

**Schema:**
```json
{
  "speakers": {
    "spk_041": {
      "consent_id": "c_260312_KP",
      "form": "A",
      "version": "1.0",
      "opt_ins": ["A1", "A3"], 
      "date": "2026-03-12",
      "re_review_date": "2026-09-12",
      "withdrawn": false,
      "withdrawn_date": null,
      "metadata": {
        "name": "...", 
        "contact": "..."
      }
    }
  }
}
```

### 2.2 The Revocation Workflow (`revocation.py`)
A script to handle the "Right to Erasure" (DPDP §11–13).

**Logic:**
1. Receive withdrawal request (email/form).
2. Lookup `speaker_anon` in `consent_register.json`.
3. Set `withdrawn = true` and `withdrawn_date = now()`.
4. Trigger `data_purge.py`:
    - Scan all `.parquet` manifests.
    - Remove all rows where `speaker_anon == target`.
    - Re-generate manifests.
5. Log action in `dataset_changelog.md`.

### 2.3 Privacy Note Injection
A set of Markdown snippets to be injected into:
- `README.md` (Global)
- `datasheet.md` (HuggingFace)
- Streamlit App UI (Footer)

## 3. Implementation Deep-Dive

### 3.1 Data Integrity
- The register must be stored on the `gpu-master` and one offline backup.
- Only the Corpus Lead has the decryption key for the `metadata` field.

### 3.2 Manifest Filtering Logic
The `manifest.py` in the TeleChannel pipeline must apply this filter:
`df = df[df['speaker_anon'].map(lambda x: not register[x].withdrawn)]`

## 4. Verification Plan

| Test Case | Input | Expected Outcome |
|---|---|---|
| **Registration** | Valid Form A | Entry created in JSON; `speaker_anon` generated. |
| **Opt-in Check** | `A2` not checked | Manifest row for that clip marked `license: internal`. |
| **Revocation** | Withdrawal request | `withdrawn` flag set $\rightarrow$ clip disappears from next manifest export. |
| **DPDP Audit** | Privacy Note check | All 3 targets (README, HF, UI) contain the DPDP disclaimer. |
