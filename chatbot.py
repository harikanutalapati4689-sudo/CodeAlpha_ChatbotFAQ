import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

nltk.download('punkt')
nltk.download('punkt_tab')

# Sample FAQ dataset (questions and answers)
faq_data = {
    "What is CodeAlpha?": "CodeAlpha is an organization that provides virtual internships in fields like AI, Web Development, and Data Science.",
    "How do I apply for an internship?": "You can apply for an internship through the official CodeAlpha website by filling out the application form.",
    "What is the duration of the internship?": "The internship duration is typically 4 weeks, but it may vary based on the program.",
    "Is the internship paid?": "CodeAlpha internships are unpaid, but they provide certificates and Letters of Recommendation.",
    "How will I receive my certificate?": "Certificates are issued after successful completion of at least 2 tasks and are sent via email.",
    "What skills do I need for the AI internship?": "Basic knowledge of Python, machine learning concepts, and libraries like scikit-learn, NLTK, or TensorFlow are helpful.",
    "Can I complete tasks at my own pace?": "Yes, you can complete the tasks at your own pace within the internship period.",
    "How do I submit my completed tasks?": "Tasks are submitted through a Google Form shared in your respective WhatsApp group.",
    "What happens if I submit only one task?": "Submitting only one task is considered incomplete, and no certificate will be issued in that case.",
    "Can I contact support if I have doubts?": "Yes, you can reach out via WhatsApp or email provided by CodeAlpha for any queries."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

def preprocess(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_best_answer(user_question):
    processed_input = preprocess(user_question)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, question_vectors)
    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I couldn't find a relevant answer. Please contact support for more help."
    
    return answers[best_match_index]

def chatbot():
    print("FAQ Chatbot - Ask me anything about CodeAlpha Internship!")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Thank you! Goodbye.")
            break
        response = get_best_answer(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chatbot()
