import unicodedata
import re

def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance giữa 2 list (tokens hoặc chars)."""
    m, n = len(a), len(b)
    # Dùng 2 hàng để tiết kiệm bộ nhớ
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Chuẩn hóa Unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Loại bỏ dấu xuống dòng (\n, \r) và thay bằng khoảng trắng
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 3. Loại bỏ ký tự đặc biệt: Chỉ giữ lại chữ cái (a-z, A-Z), 
    # chữ có dấu (Vietnamese), số (0-9) và khoảng trắng.
    # Regex [^\w\s] nghĩa là: "Cái gì KHÔNG PHẢI là chữ/số và khoảng trắng thì xóa"
    # Nhưng để an toàn với tiếng Việt, ta dùng cách liệt kê "ngược" để xóa dấu câu:
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    
    # 4. Bỏ khoảng trắng thừa ở giữa, đầu và cuối
    return " ".join(text.split()).strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(reference_chars)."""
    ref = list(normalize_text(reference))
    hyp = list(normalize_text(hypothesis))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = edit_distance(words) / len(reference_words)."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def compute_metrics(reference: str, hypothesis: str) -> dict:
    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "wer_pct": round(wer * 100, 2),
        "cer_pct": round(cer * 100, 2),
    }

if __name__ == "__main__":
    # Test nhanh
    ref = "##Tôi là minh hoàng   . Hêllo \nworld!!"
    hyp = "Tôi là minh hoàng."
    metrics = compute_metrics(ref, hyp)

    print(normalize_text(ref))
    print(f"WER: {metrics['wer_pct']}%  CER: {metrics['cer_pct']}%")
