# Implementation Backlog: Consent & Privacy (MOD-A)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-A-01 | Create Consent Register | Implement `consent_register.json` using the schema in TDD_MOD_A_01. | JSON file exists with at least one test entry. | None |
| MOD-A-02 | Build Revocation Script | Create `revocation.py` to flip `withdrawn` flag and trigger purge. | Script successfully removes a test speaker from a manifest. | MOD-A-01 |
| MOD-A-03 | Implement Manifest Filter | Update `telechannel/manifest.py` to filter out withdrawn speakers. | `manifest.py` excludes speakers marked `withdrawn=true`. | MOD-A-01 |
| MOD-A-04 | Deploy Privacy Notes | Inject DPDP notes into README, HF datasheet, and Streamlit UI. | All three locations verified to have the exact legal text. | None |
| MOD-A-05 | Establish Consent Audit | Create a simple process for logging paper forms $\rightarrow$ JSON entries. | 100% of recorded clips traceable to a `consent_id`. | MOD-A-01 |
