# PhiStream v1 fixture storage format

A **directory** contains exactly these two required files:

- `manifest.json`: `format = PHISTREAM_V1_F32_LE`, config, tensor records (names, shape, `<f4` dtype, byte offset and size, SHA256), seed and simulated provenance.
- `weights.bin`: all dense FP32 tensors in little-endian C row-major layout. Tensor offsets are padded to 4096-byte boundaries and recorded in manifest. Bytes are mmap'ed read-on-demand.

The loader validates strict tensor-name/shape agreement, bounds and non-overlapping regions. The optional `verify` command validates the entire contents against per-tensor SHA256. Input is local; no pickle/code execution occurs. Manifest size and dimensions are *not* hardened against adversarial huge untrusted input. **Do not open checkpoints supplied by unknown parties** in the current research version.

These are weights for one specific experimental decoder architecture. Translating a trained Llama/Qwen model requires a separate, correctly tested architecture adapter and authorization under the model license. The fixture isn't trained and cannot hold a conversation.
