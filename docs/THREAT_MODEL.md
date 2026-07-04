# Threat Model

> **Status: draft.** Planned sections outlined below.

Planned sections:

1. **Cross-lingual cache poisoning (headline).** Attacker seeds an entry in
   language X; a victim in language Y is served it and is less able to detect
   tampering. Mitigations: structural guard, per-pair precision-tuned
   thresholds, tenant scoping, audit trail, TTL rotation.
2. **Tenant isolation & the hit-rate/leak tension.** Cross-tenant sharing
   drives value *and* leak surface — stated honestly, with an opt-out.
3. **Timing side-channel.** Constant-time padding scoped **only** to shared
   semantic hits (not blanket). Random jitter is *not* a fix.
4. **Data at rest.** AES-256-GCM; pluggable KeyProvider (env-var vs KMS/Vault
   envelope). Honest boundary on what env-var mode does and does not protect.
5. **Cache flooding / noisy neighbour (DoS).** Per-tenant byte quota +
   token-count rate limiting.
6. **Horizontal-scaling caveat.** In-memory quota/limiter don't share state
   across instances → multi-node needs Redis. Stated, not hidden.
7. **Prompt/response confidentiality.** Zero plaintext in audit logs; payload
   logging opt-in + encrypted + separate stream.
