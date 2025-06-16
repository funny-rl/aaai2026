import json

def check_duplicate_name_value(file_path):
    name_values = set()  # 'name' 값들의 고유한 집합을 저장
    duplicate_name_values = set() # 중복되는 'name' 값들을 저장

    print(f"Checking for duplicate 'name' values in '{file_path}'...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                if 'name' in data:
                    name_value = data['name']
                    if name_value in name_values:
                        duplicate_name_values.add(name_value)
                        # 중복 발견 시 즉시 경고 메시지 출력
                        print(f"  Warning: Duplicate 'name' value found: '{name_value}' on line {line_num}")
                    else:
                        name_values.add(name_value)
                else:
                    # 'name' 키가 없는 경우 경고 메시지 출력
                    print(f"  Warning: 'name' key not found in line {line_num}: {line.strip()}")
            except json.JSONDecodeError as e:
                # JSON 파싱 오류 발생 시 메시지 출력
                print(f"  Error decoding JSON on line {line_num}: {e} - {line.strip()}")
    
    print("\n--- Summary of Duplicate 'name' Values ---")
    if duplicate_name_values:
        for value in duplicate_name_values:
            print(f"- **'{value}'** (appears multiple times)")
        print(f"\n{len(duplicate_name_values)} unique duplicate 'name' values were found.")
        return True
    else:
        print("No duplicate 'name' values found.")
        return False

file_to_check = 'CoT.jsonl' 

if __name__ == "__main__":
    check_duplicate_name_value(file_to_check)
