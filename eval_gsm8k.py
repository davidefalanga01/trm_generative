import re
from typing import Optional

class GSM8KEvaluator:
    def __init__(self):
        # Handle various number formats, including commas, decimals, and optional signs
        self.number_pattern = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")

    def _clean_number(self, text: str) -> str:
        """
        Clean text to isolate the pure number.
        Drop commas, spaces, symbols $ or %, and convert in string.
        """
        if not text:
            return ""
        
        # 1. Find the last number in the text
        matches = self.number_pattern.findall(text)
        if not matches:
            return ""
        
        last_num = matches[-1]
        
        # 2. Undo formatting ("1,200" -> "1200")
        clean_num = last_num.replace(",", "").replace("$", "").replace("%", "")
        
        # 3. Handle float: "42.00" deve essere uguale a "42"
        try:
            val = float(clean_num)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except ValueError:
            return clean_num

    def extract_answer(self, completion: str) -> str:
        """
        Extract the final answer from the model's completion, handling various formats.
        """
        # Case 1: The model have to use GSM8K standard format (#### answer)
        if "####" in completion:
            return self._clean_number(completion.split("####")[-1])
        
        # Case 2: The model use LaTeX boxed format (\boxed{answer})
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", completion)
        if boxed_match:
            return self._clean_number(boxed_match.group(1))
            
        # Case 3: Fallback (Euristica "Last Number")
        return self._clean_number(completion)

    def is_correct(self, model_output: str, ground_truth: str) -> bool:
        """
        Compare the model's output with the ground truth, extracting and normalizing the answer from both.
        """
        # Ground truth in GSM8K "steps of reasoning #### answer"
        gt_clean = self.extract_answer(ground_truth)
        pred_clean = self.extract_answer(model_output)
        
        print(f"Confronto: Pred='{pred_clean}' vs GT='{gt_clean}'")
        
        return pred_clean == gt_clean

if __name__ == "__main__":
    evaluator = GSM8KEvaluator()

    pred1 = "Calcolo 50 + 50. La risposta è 100."
    gt1 = "50+50=100 #### 100"
    print(f"Test 1: {evaluator.is_correct(pred1, gt1)}") # True
    
    pred2 = "Il totale è $1,250.00"
    gt2 = "#### 1250"
    print(f"Test 2: {evaluator.is_correct(pred2, gt2)}") # True 
    
    pred3 = "Penso sia 42."
    gt3 = "#### 43"
    print(f"Test 3: {evaluator.is_correct(pred3, gt3)}") # False
