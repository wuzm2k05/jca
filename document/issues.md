# Issues
## deepseek API 风控 t1.0.0
1. OpenAIInvalidRequestError: Error code: 400 - {'error': {'message': 'Your input contains content restricted by our safety guidelines. Please ensure your prompt does not involve: (1) political/social sensitive topics, (2) illegal/criminal acts, (3) explicit/vulgar content, or (4) cybersecurity or technical integrity violations. (request id: 7a4ac8a9aac7c400a9e4f341dc94e91b)', 'type': 'api_error', 'param': '', 'code': 'sensitive_words_detected'}}
1. llm会在任意目录修改文件
