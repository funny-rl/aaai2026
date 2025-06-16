import json
from transformers import AutoTokenizer

def find_longest_reason_value(file_path, tokenizer_name="klue/bert-base"):
    """
    JSONL 파일에서 'reason' 키의 값이 가장 긴 row를 찾아 
    해당 값과 HF 토큰 수를 반환합니다.
    """
    
    # 토크나이저 로드
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Tokenizer loaded successfully.")

    max_length = -1
    longest_reason_value = ""
    longest_reason_row = None
    
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if "reason" in data and isinstance(data["reason"], str):
                    current_reason = data["reason"]
                    # 토크나이저를 사용하여 토큰화하고 길이 계산
                    tokens = tokenizer.tokenize(current_reason)
                    current_length = len(tokens)

                    if current_length > max_length:
                        max_length = current_length
                        longest_reason_value = current_reason
                        longest_reason_row = data
                elif "reason" not in data:
                    print(f"Warning: 'reason' key not found in line {line_num}.")
                elif not isinstance(data["reason"], str):
                    print(f"Warning: 'reason' value is not a string in line {line_num}. Value: {data['reason']}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num}: {e} - Line content: {line.strip()}")
            except Exception as e:
                print(f"An unexpected error occurred on line {line_num}: {e} - Line content: {line.strip()}")

    if longest_reason_row:
        print("\n--- Result ---")
        print(f"Longest 'reason' value found (HF Tokens: {max_length}):")
        print(f"Value: \"{longest_reason_value}\"")
        print(f"Full row data: {json.dumps(longest_reason_row, ensure_ascii=False, indent=2)}")
    else:
        print("\nNo 'reason' key with string value found in the file.")

    return longest_reason_value, max_length, longest_reason_row

# train.jsonl 파일 경로를 지정해 주세요.
file_path = './valid.jsonl'

if __name__ == "__main__":
    # 실행하기 전에 'transformers' 라이브러리가 설치되어 있는지 확인하세요:
    # pip install transformers
    longest_value, token_count, row_data = find_longest_reason_value(file_path)
