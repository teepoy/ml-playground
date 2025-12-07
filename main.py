import warnings


def main():
    # Display security warning
    warnings.warn(
        "\n" + "="*70 + "\n"
        "SECURITY WARNING: This project uses PyTorch 1.13.1\n"
        "which has known security vulnerabilities:\n"
        "  - Heap buffer overflow (CVE pending)\n"
        "  - Use-after-free vulnerability (CVE pending)\n"
        "  - RCE via torch.load (CVE pending)\n"
        "\n"
        "DO NOT use in production without upgrading to PyTorch 2.6.0+\n"
        "See SECURITY.md for details and mitigation strategies.\n"
        + "="*70,
        RuntimeWarning,
        stacklevel=2
    )
    print("Hello from ml-playground!")


if __name__ == "__main__":
    main()
