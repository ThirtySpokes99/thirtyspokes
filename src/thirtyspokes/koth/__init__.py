"""The library the v3 mechanism (`v3/`) and the serving path (`serve/`) import.

What is here survived the 2026-09-07 cutover because v3 imports it: real Intel TDX quotes and
their full DCAP verification (`tdx`, `collateral`), enclave-sealed keys (`sealed`), the
OpenRouter key gate (`orkey`), the chained manifest the schedule root is built from
(`holdout_feed`), corpus freshness (`corpus`), the pinned embedding harness and routing pool
(`harness`), reference-record signing (`reference`), image records (`imagestore`) and on-chain
measurement governance (`governance`). The KOTH-TEE mechanism that once lived in this package —
miners running benchmarks in their own TEE, validators verifying a proof — was removed with v2.
"""
