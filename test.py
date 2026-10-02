import anthropic

client = anthropic.Anthropic(
    base_url="http://10.10.50.40",
    api_key="sk-OkWwm45JQ3jbjcwkzcNX9YxStb31H5iqXxkoGpMU6vYkFrEg",
)

message = client.messages.create(
    model="deepseek-v4-flash",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain quantum entanglement in one paragraph."}],
)

print(message.content[0].text)
