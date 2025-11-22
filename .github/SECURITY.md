# Security Policy

## Supported Versions

We currently support the following versions of this project:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do not** open a public issue
2. Email us at security@deepfake-forensics.com
3. Include as much detail as possible about the vulnerability
4. We will respond within 48 hours

## Security Considerations

### Data Privacy
- This project processes images and videos that may contain sensitive content
- We do not store or transmit user data by default
- All processing is done locally unless explicitly configured otherwise

### Model Security
- Models may be vulnerable to adversarial attacks
- Use appropriate defenses and validation
- Regularly update models and dependencies

### API Security
- API endpoints should be secured with authentication
- Rate limiting is recommended for production use
- Input validation is enforced on all endpoints

## Responsible Disclosure

We follow responsible disclosure practices:
- Vulnerabilities are kept confidential until patched
- We will credit researchers who report vulnerabilities
- We aim to patch critical vulnerabilities within 30 days

## Contact

For security-related questions or concerns:
- Email: security@deepfake-forensics.com
- PGP Key: [Available upon request]
