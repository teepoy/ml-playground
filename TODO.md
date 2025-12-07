# TODO List

## ✅ Security Update Complete

### PyTorch Upgrade (COMPLETED)

**Status**: ✅ Complete
**Previous Version**: PyTorch 1.13.1 (vulnerable)
**Current Version**: PyTorch 2.9.1 (secure)

All security vulnerabilities have been addressed by upgrading to PyTorch 2.9.1.

### Submodule Compatibility (NEW PRIORITY)

**Status**: Needs Verification
**Priority**: HIGH

The mmpretrain and mmdetection submodules were originally tested with PyTorch 1.x. They may require updates to work with PyTorch 2.9+:

**Action Items:**
1. Test mmpretrain compatibility with PyTorch 2.9
2. Test mmdetection compatibility with PyTorch 2.9
3. Update submodules to newer versions if needed
4. Document any compatibility issues or workarounds

---

## Future Enhancements

### Short Term
- [ ] Add comprehensive test suite
- [ ] Set up CI/CD pipeline
- [ ] Add linting and code quality checks
- [ ] Configure pre-commit hooks

### Medium Term
- [ ] Add example scripts for using mmpretrain
- [ ] Add example scripts for using mmdetection
- [ ] Create Jupyter notebook tutorials
- [ ] Add Docker support for isolated environments

### Long Term
- [ ] Migrate to secure PyTorch version (2.6.0+)
- [ ] Update all documentation post-upgrade
- [ ] Add performance benchmarks
- [ ] Create comprehensive developer guide

---

## Notes

- **DO NOT** use this project in production with current PyTorch version
- Always load models from trusted sources only
- See SECURITY.md for temporary mitigation strategies
