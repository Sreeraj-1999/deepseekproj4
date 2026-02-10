import json
import re

# Load your dataset
input_path = r"C:\Users\User\Desktop\siemens\OFFSHORE\converted_alpaca3_reasoning_gpt.json"  # Update this path
output_path = "CLEAN-converted_alpaca3_reasoning_gpt_CLEAN.json"

def clean_html(text):
    """Remove all HTML tags and artifacts from text"""
    if not text:
        return text
    
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', '\n', text)  # <br> to newline
    text = re.sub(r'<[^>]+>', '', text)       # Remove all other HTML tags
    
    # Remove HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    
    # Remove markdown bold/italic but keep text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    # Remove \boxed{...} LaTeX
    text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def clean_output_field(text):
    """Clean the output field, preserving <think> blocks"""
    if not text:
        return text
    
    # Split into think and answer parts
    if '<think>' in text and '</think>' in text:
        think_start = text.index('<think>')
        think_end = text.index('</think>') + len('</think>')
        
        think_block = text[think_start:think_end]
        answer_part = text[think_end:]
        
        # Clean inside think block (between tags)
        think_content = think_block[len('<think>'):-len('</think>')]
        think_content = clean_html(think_content)
        
        # Clean answer part
        answer_part = clean_html(answer_part)
        
        return f"<think>\n{think_content}\n</think>\n{answer_part}"
    else:
        return clean_html(text)

# Load dataset
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Track changes
html_found = 0
samples_with_html = []

for i, sample in enumerate(data):
    original_output = sample.get('output', '')
    
    # Check if HTML exists
    if re.search(r'<(?!think|/think)[^>]+>', original_output):
        html_found += 1
        if html_found <= 5:  # Show first 5 examples
            samples_with_html.append({
                'index': i,
                'instruction': sample.get('instruction', '')[:80],
                'html_snippet': re.findall(r'<(?!think|/think)[^>]+>', original_output)[:5]
            })
    
    # Clean all fields
    sample['instruction'] = clean_html(sample.get('instruction', ''))
    sample['input'] = clean_html(sample.get('input', ''))
    sample['output'] = clean_output_field(sample.get('output', ''))

print(f"\nSamples with HTML tags in output: {html_found}/{len(data)} ({html_found/len(data)*100:.1f}%)")
print("\nExamples of HTML found:")
for s in samples_with_html:
    print(f"  Sample {s['index']}: {s['instruction']}")
    print(f"    HTML tags: {s['html_snippet']}")

# Save cleaned dataset
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nCleaned dataset saved to: {output_path}")
print("Use this file for your next training run.")