# Security Notice

## Known Vulnerabilities in PyTorch 1.13.1

This project currently uses PyTorch 1.13.1 as specified in the project requirements. However, this version has several known security vulnerabilities:

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
- **Description**: PyTorch deserialization vulnerability (details withdrawn).
- **Mitigation**: Exercise caution when deserializing PyTorch objects from untrusted sources.

## Recommendations

### For Production Use
**⚠️ CRITICAL**: Do not use PyTorch 1.13.1 in production environments. Upgrade to PyTorch 2.6.0 or later.

To upgrade:
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

1. Test compatibility with PyTorch 2.2.0+ first (minimum for basic security)
2. Update mmpretrain and mmdetection submodules to versions compatible with newer PyTorch
3. Adjust Python version requirement to support PyTorch 2.6.0+ (requires Python 3.8-3.12)
4. Run comprehensive tests to ensure functionality is maintained

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
