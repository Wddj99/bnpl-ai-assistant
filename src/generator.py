from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AnswerGenerator:

    def __init__(self):
        model_name = "google/flan-t5-base"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question, context):

        prompt = f"""
You are a BNPL customer support assistant.

Answer the customer's question using ONLY the information in the context.

IMPORTANT RULES:
1. Give a complete answer, not just "Yes" or "No".
2. Use the information that directly answers the question.
3. Do not combine unrelated information from different topics.
4. Do not invent information.
5. If the context does not contain enough information, say that the available information is insufficient.
6. Keep the answer concise and customer-friendly.

Context:
{context}

Customer question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()