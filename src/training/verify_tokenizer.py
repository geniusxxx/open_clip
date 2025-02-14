def verify_tokenizer():
    """验证tokenizer的词表和特殊token"""
    from open_clip import get_tokenizer
    
    # 获取tokenizer
    tokenizer = get_tokenizer('MobileCLIP-S1')
    
    # 1. 打印词表大小
    print(f"词表大小: {len(tokenizer.encoder)}")
    
    # 2. 检查特殊token
    print("\n特殊Token:")
    special_tokens = {
        'PAD': 0,
        'EOT': 49407
    }
    for name, id in special_tokens.items():
        token = next((k for k, v in tokenizer.encoder.items() if v == id), None)
        print(f"{name} token: {token} (ID: {id})")
    
    # 3. 测试一个简单的句子
    text = "Hello world!"
    tokens = tokenizer(text)
    print(f"\n测试句子: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"最大token ID: {max(tokens)}")
    print(f"是否为EOT: {max(tokens) == 49407}")

if __name__ == "__main__":
    verify_tokenizer()
