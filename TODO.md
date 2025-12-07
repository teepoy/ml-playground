# TODO List

## 🚨 Critical Security Issue

### Upgrade PyTorch 1.13.1 (HIGH PRIORITY)

**Status**: Pending  
**Priority**: CRITICAL  
**Blocked by**: Submodule compatibility verification needed

PyTorch 1.13.1 has 4 known security vulnerabilities including a **CRITICAL** remote code execution vulnerability. See [SECURITY.md](SECURITY.md) for full details.

**Before upgrading:**
1. Verify mmcv 2.x supports PyTorch 2.6+
2. Check mmpretrain/mmdetection compatibility
3. Test in isolated environment
4. Update submodules if needed

**Issue Template**: `.github/ISSUE_TEMPLATE/security-pytorch-upgrade.md`

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
