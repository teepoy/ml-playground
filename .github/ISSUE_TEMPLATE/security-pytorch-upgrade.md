---
name: Security Issue - Upgrade PyTorch 1.13.1
about: Track the upgrade of PyTorch 1.13.1 to address security vulnerabilities
title: '[SECURITY] Upgrade PyTorch 1.13.1 to address critical vulnerabilities'
labels: security, enhancement, dependencies
assignees: ''
---

## Security Issue: PyTorch 1.13.1 Vulnerabilities

### Priority: HIGH / CRITICAL

This project currently uses **PyTorch 1.13.1** which has **4 known security vulnerabilities**:

### Vulnerabilities

1. **Heap Buffer Overflow Vulnerability**
   - Severity: High
   - Affected versions: < 2.2.0
   - Patched version: 2.2.0

2. **Use-After-Free Vulnerability**
   - Severity: High  
   - Affected versions: < 2.2.0
   - Patched version: 2.2.0

3. **Remote Code Execution via torch.load** ⚠️ **CRITICAL**
   - Severity: Critical
   - Affected versions: < 2.6.0
   - Patched version: 2.6.0
   - **Risk**: Arbitrary code execution when loading untrusted model files

4. **Deserialization Vulnerability**
   - Severity: High
   - Affected versions: <= 2.3.1
   - Advisory Status: Withdrawn

### Current Constraints

The upgrade is blocked by submodule compatibility requirements:

- **mmpretrain** requires: `mmcv>=2.0.0,<2.4.0` and `mmengine>=0.8.3,<1.0.0`
- **mmdetection** requires: `mmcv>=2.0.0rc4,<2.2.0` and `mmengine>=0.7.1,<1.0.0`

Need to verify if these mmcv/mmengine versions support PyTorch 2.6.0+.

### Action Items

- [ ] Research mmcv 2.x compatibility with PyTorch 2.6+
- [ ] Check if newer versions of mmpretrain/mmdetection support PyTorch 2.6+
- [ ] Test PyTorch 2.6.0 with current submodule versions in isolated environment
- [ ] Update submodules to compatible versions (if available)
- [ ] Update Python version requirement to 3.9-3.12
- [ ] Update pyproject.toml to use PyTorch 2.6.0+
- [ ] Run full test suite to verify compatibility
- [ ] Update documentation to remove security warnings

### Temporary Mitigations (Until Upgrade)

⚠️ **DO NOT use this project in production with PyTorch 1.13.1**

If you must use it:
- Only load models from trusted, verified sources
- Never use with untrusted input data
- Isolate in containers/VMs
- Keep network isolated from untrusted sources
- Regular security audits

### Target Upgrade

```toml
dependencies = [
    "torch>=2.6.0",
    "torchvision>=0.19.0",
    "numpy>=1.26.0",
]
requires-python = ">=3.9,<3.13"
```

### References

- See [SECURITY.md](../../SECURITY.md) for complete details
- PyTorch Security Advisories: https://github.com/pytorch/pytorch/security/advisories
- mmcv documentation: https://github.com/open-mmlab/mmcv
- mmpretrain documentation: https://github.com/open-mmlab/mmpretrain
- mmdetection documentation: https://github.com/open-mmlab/mmdetection

### Additional Context

This issue was created automatically as a reminder. The project was initially set up with PyTorch 1.13.1 per requirements, but upgrade should be prioritized before any production use.
