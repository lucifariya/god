
# Program for Question 1: Caesar Cipher Implementation

def caesar_cipher(text, shift):
    result = ''
    for char in text:
        if char.isalpha():
            shift_base = 65 if char.isupper() else 97
            result += chr((ord(char) - shift_base + shift) % 26 + shift_base)
        else:
            result += char
    return result

def question_1():
    input_string = input("Enter a string with special characters, spaces, and numbers: ")
    filtered_string = ''.join([char for char in input_string if char.isalpha()])
    print("Filtered string (only alphabets):", filtered_string)
    shift_value = 3
    encrypted_text = caesar_cipher(filtered_string, shift_value)
    print("Encrypted text:", encrypted_text)
    decrypted_text = caesar_cipher(encrypted_text, -shift_value)
    print("Decrypted text:", decrypted_text)

# Program for Question 2: Hotel Feedback Survey

def question_2():
    feedback_scores = {"excellent": 5, "good": 4, "ok": 3, "compromised": 2, "poor": 1}
    n = int(input("Enter the number of people giving feedback: "))
    feedback = []
    for _ in range(n):
        feedback_input = input(f"Enter feedback ({', '.join(feedback_scores.keys())}): ").lower()
        feedback.append(feedback_scores.get(feedback_input, 0))
    excellent_or_good = sum(1 for score in feedback if score >= 4)
    ok = sum(1 for score in feedback if score == 3)
    excellent_or_good_percentage = (excellent_or_good / n) * 100
    ok_percentage = (ok / n) * 100
    if excellent_or_good_percentage > 75:
        print("3-star hotel")
    elif excellent_or_good_percentage > 50 and ok_percentage > 25:
        print("2-star hotel")
    elif excellent_or_good_percentage + ok_percentage > 75:
        print("1-star hotel")
    else:
        print("No-star hotel")
    