---
name: Verify Submodule Compatibility with PyTorch 2.9
about: Test and verify that mmpretrain and mmdetection work with upgraded PyTorch
title: '[TASK] Verify mmpretrain and mmdetection compatibility with PyTorch 2.9'
labels: enhancement, testing, submodules
assignees: ''
---

## Submodule Compatibility Verification

### Priority: HIGH

This project has been **upgraded from PyTorch 1.13.1 to PyTorch 2.9.1** to address critical security vulnerabilities. The mmpretrain and mmdetection submodules need to be tested for compatibility with the new PyTorch version.

### Background

**Security Upgrade Completed:**
- ✅ PyTorch 1.13.1 → 2.9.1
- ✅ All security vulnerabilities resolved
- ✅ torchvision upgraded to 0.24.1

**Submodule Requirements:**
- **mmpretrain** requires: `mmcv>=2.0.0,<2.4.0` and `mmengine>=0.8.3,<1.0.0`
- **mmdetection** requires: `mmcv>=2.0.0rc4,<2.2.0` and `mmengine>=0.7.1,<1.0.0`

These submodules were originally designed for PyTorch 1.x and need compatibility verification with PyTorch 2.9.

### Action Items

- [ ] **Test mmpretrain** with PyTorch 2.9.1
  - [ ] Install required dependencies (mmcv, mmengine)
  - [ ] Run basic import tests
  - [ ] Test with sample models
  - [ ] Document any errors or issues

- [ ] **Test mmdetection** with PyTorch 2.9.1
  - [ ] Install required dependencies (mmcv, mmengine)
  - [ ] Run basic import tests
  - [ ] Test with sample models
  - [ ] Document any errors or issues

- [ ] **Check for updates**
  - [ ] Check if newer versions of mmpretrain support PyTorch 2.9
  - [ ] Check if newer versions of mmdetection support PyTorch 2.9
  - [ ] Update submodules if compatible versions exist

- [ ] **Document findings**
  - [ ] Update README.md with compatibility status
  - [ ] Add usage examples if working
  - [ ] Document workarounds if needed
  - [ ] Update SECURITY.md if necessary

### Testing Steps

1. **Install submodule dependencies:**
   ```bash
   cd packages/mmpretrain
   pip install -e .
   # Test basic functionality
   ```

2. **Check mmcv compatibility:**
   ```bash
   pip install "mmcv>=2.0.0,<2.2.0"  # Most restrictive constraint
   python -c "import mmcv; print(mmcv.__version__)"
   ```

3. **Run compatibility tests:**
   - Import checks
   - Basic model loading
   - Sample inference

### Expected Outcomes

✅ **Best case**: Submodules work with PyTorch 2.9 without modification
⚠️ **Likely case**: Need to update submodules to newer versions
❌ **Worst case**: Need to pin specific mmcv/mmengine versions and document limitations

### References

- See [SECURITY.md](../../SECURITY.md) for complete details
- PyTorch Security Advisories: https://github.com/pytorch/pytorch/security/advisories
- mmcv documentation: https://github.com/open-mmlab/mmcv
- mmpretrain documentation: https://github.com/open-mmlab/mmpretrain
- mmdetection documentation: https://github.com/open-mmlab/mmdetection

### Additional Context

This issue was created automatically as a reminder. The project was initially set up with PyTorch 1.13.1 per requirements, but upgrade should be prioritized before any production use.
