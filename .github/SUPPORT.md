# Support

## Getting Help

### Documentation
- [Project Documentation](https://deepfake-forensics.readthedocs.io)
- [API Reference](https://deepfake-forensics.readthedocs.io/en/latest/api/)
- [Examples](https://deepfake-forensics.readthedocs.io/en/latest/examples/)

### Community
- [GitHub Discussions](https://github.com/deepfake-forensics/deepfake-forensics/discussions)
- [Discord Server](https://discord.gg/deepfake-forensics)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/deepfake-forensics)

### Professional Support
- Email: support@deepfake-forensics.com
- Enterprise Support: enterprise@deepfake-forensics.com

## Reporting Issues

### Bug Reports
- Use the [bug report template](https://github.com/deepfake-forensics/deepfake-forensics/issues/new?template=bug_report.md)
- Include system information and error logs
- Provide steps to reproduce the issue

### Feature Requests
- Use the [feature request template](https://github.com/deepfake-forensics/deepfake-forensics/issues/new?template=feature_request.md)
- Describe the use case and expected behavior
- Consider contributing the feature yourself

### Security Issues
- Email: security@deepfake-forensics.com
- Do not open public issues for security vulnerabilities
- See [SECURITY.md](SECURITY.md) for more information

## FAQ

### Installation Issues
**Q: I'm getting import errors after installation.**
A: Make sure you have Python 3.11+ and all dependencies are installed. Try:
```bash
pip install -e ".[dev]"
```

**Q: CUDA/GPU support not working.**
A: Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Usage Issues
**Q: Model predictions are not accurate.**
A: Ensure you're using the correct model for your data type and that input images are properly preprocessed.

**Q: API server won't start.**
A: Check that the model checkpoint exists and is compatible with the current version.

### Performance Issues
**Q: Inference is slow.**
A: Try using GPU acceleration, reducing batch size, or using a smaller model.

**Q: Training is taking too long.**
A: Consider using a smaller dataset for testing, reducing model complexity, or using mixed precision training.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

