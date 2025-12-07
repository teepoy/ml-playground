# Security Notice

## ✅ Security Status: RESOLVED

This project has been **upgraded to PyTorch 2.9.1** which addresses all previously identified security vulnerabilities.

## Previously Identified Vulnerabilities (Now Fixed)

The project was originally set up with PyTorch 1.13.1 which had several known security vulnerabilities. These have been resolved by upgrading to PyTorch 2.9.1:

### 1. Heap Buffer Overflow Vulnerability
- **Severity**: High
- **Affected versions**: < 2.2.0
- **Patched version**: 2.2.0
- **Description**: PyTorch contains a heap buffer overflow vulnerability that could lead to memory corruption.
- **Mitigation**: Upgrade to PyTorch 2.2.0 or later.

### 2. Use-After-Free Vulnerability
- **Severity**: High
- **Affected versions**: < 2.2.0
- **Patched version**: 2.2.0
- **Description**: PyTorch contains a use-after-free vulnerability that could be exploited for arbitrary code execution.
- **Mitigation**: Upgrade to PyTorch 2.2.0 or later.

### 3. Remote Code Execution via torch.load
- **Severity**: Critical
- **Affected versions**: < 2.6.0
- **Patched version**: 2.6.0
- **Description**: `torch.load` with `weights_only=True` can still lead to remote code execution when loading untrusted model files.
- **Mitigation**: 
  - Upgrade to PyTorch 2.6.0 or later
  - **NEVER load untrusted model files with PyTorch 1.13.1**
  - Only load models from trusted sources
  - Implement additional validation before loading any model files

### 4. Deserialization Vulnerability
- **Severity**: High
- **Affected versions**: <= 2.3.1
- **Advisory Status**: Withdrawn
- **Description**: This advisory has been withdrawn by the reporting organization. A withdrawn advisory typically means the vulnerability assessment was determined to be incorrect, a duplicate, or the issue was resolved through other means. However, general caution is still advised when deserializing PyTorch objects from untrusted sources.
- **Mitigation**: Exercise caution when deserializing PyTorch objects from untrusted sources.

## Recommendations

### For Production Use
**⚠️ CRITICAL**: Do not use PyTorch 1.13.1 in production environments. Upgrade to PyTorch 2.6.0 or later.

**Note on Compatibility**: This project includes mmpretrain and mmdetection as submodules, which have specific requirements:
- **mmpretrain** requires: `mmcv>=2.0.0,<2.4.0` and `mmengine>=0.8.3,<1.0.0`
- **mmdetection** requires: `mmcv>=2.0.0rc4,<2.2.0` and `mmengine>=0.7.1,<1.0.0`

Before upgrading PyTorch, verify that the required mmcv and mmengine versions support the newer PyTorch version. You may need to:
1. Update the submodules to newer versions that support PyTorch 2.6+
2. Or fork the submodules and update their dependencies

To upgrade (after verifying compatibility):
```bash
# Update pyproject.toml dependencies to:
dependencies = [
    "torch>=2.6.0",
    "torchvision>=0.19.0",
    "numpy>=1.26.0",
]
```

### For Development/Testing
If you must use PyTorch 1.13.1 for compatibility reasons:

1. **Isolate the environment**: Use containers or virtual machines to isolate the application
2. **Never load untrusted data**: Only load models and data from verified, trusted sources
3. **Network isolation**: Keep systems using PyTorch 1.13.1 isolated from untrusted networks
4. **Input validation**: Implement strict validation for all inputs before processing
5. **Regular security audits**: Monitor for new vulnerabilities and plan migration path

### Migration Path

When ready to upgrade to a secure version:

1. **Check submodule compatibility**: Verify that mmpretrain and mmdetection work with PyTorch 2.6+
   - Current mmpretrain requires `mmcv<2.4.0`
   - Current mmdetection requires `mmcv<2.2.0`  
   - Check if newer versions of these packages support PyTorch 2.6+
   
2. **Update submodules** (if needed):
   ```bash
   cd packages/mmpretrain && git pull origin main
   cd ../mmdetection && git pull origin main
   ```

3. **Test compatibility** with PyTorch 2.2.0+ first (minimum for basic security)

4. **Adjust Python version** requirement to support PyTorch 2.6.0+ (requires Python 3.9-3.12)

5. **Run comprehensive tests** to ensure functionality is maintained with both PyTorch and the submodules

## Security Best Practices

- **Never** use `torch.load()` on files from untrusted sources
- Always validate model file integrity before loading
- Keep dependencies updated and monitor security advisories
- Use virtual environments to isolate dependencies
- Implement defense-in-depth security measures
- Regular security scanning of dependencies

## Reporting Security Issues

If you discover additional security issues, please report them through appropriate channels rather than creating public issues.

---

**Last Updated**: 2025-12-07
