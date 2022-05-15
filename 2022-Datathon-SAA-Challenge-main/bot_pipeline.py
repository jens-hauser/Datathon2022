from haystack.nodes import PDFToTextConverter, TextConverter, PreProcessor
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever 
from haystack.nodes import TransformersReader, FARMReader
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from subprocess import Popen, PIPE, STDOUT
from typing import Callable
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()

APP_TOKEN = os.getenv('APP_TOKEN')
BOT_TOKEN = os.getenv('BOT_TOKEN')


filenames = [
    "Travel Insurance FAQs Allianz Global Assistance.json",
    "COVID-19 FAQs Allianz Global Assistance.json",
    "allianz-ch-travel-faq-en-kendra.json",
]


document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
doc_pdf_1 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_Cancellation_charges_for_accommodation_and_further_training_01-2022.pdf", meta={"name": "cancel"})[0]
doc_pdf_2 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_SecureTrip_Budget_01-2022.pdf", meta={"name": "budget"})[0]
doc_pdf_3 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_SecureTrip_Classic_01-2022.pdf", meta={"name": "classic"})[0]
doc_pdf_4 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_SecureTrip_PremiumPLUS_01-2022.pdf", meta={"name": "premiumplus"})[0]
doc_pdf_5 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_SecureTrip_Premium_01-2022.pdf", meta={"name": "premium"})[0]
doc_pdf_6 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_Travel_protection_package_Classic_01-2022.pdf", meta={"name": "classic"})[0]
doc_pdf_7 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_Travel_protection_package_PremiumPLUS_01-2022.pdf", meta={"name": "premiumplus"})[0]
doc_pdf_8 = converter.convert(file_path="dataset/terms-and-conditions/AzTrv_GTC_Travel_protection_package_Premium_01-2022.pdf", meta={"name": "premium"})[0]

docs = [doc_pdf_1, doc_pdf_2, doc_pdf_3, doc_pdf_4, doc_pdf_5, doc_pdf_6, doc_pdf_7, doc_pdf_8]

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=1000,
    split_respect_sentence_boundary=True,
)
docs_preprocessed = []
for doc in docs:
    docs_preprocessed += preprocessor.process([doc])

    
for file_name in filenames:
    text = ""
    with open(f"dataset/faq/{file_name}") as json_file:
        faq_data = json.load(json_file)["FaqDocuments"]
        for qa_pair in faq_data:
            text += qa_pair["Question"] + ": " + qa_pair["Answer"] + "\n"
            
        doc =  {
        'content': text,
        'meta': {'name': "faq"}
        }   
            
        docs_preprocessed.append(doc)
        
document_store.write_documents(docs_preprocessed)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    max_seq_len_query=64,
    max_seq_len_passage=256,
    batch_size=16,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
)

document_store.update_embeddings(retriever)

generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1')

def _create_dict_with_faqs(filenames):
    faq_data_total = {}
    for file_name in filenames:
        # Opening JSON file
        with open(f"dataset/faq/{file_name}") as json_file:
            faq_data = json.load(json_file)["FaqDocuments"]
            faq_data = {faq_pair["Question"]: faq_pair["Answer"] for faq_pair in faq_data}
            faq_data_total = {**faq_data_total, **faq_data}
    return faq_data_total

def check_for_similar_qa(question, filenames, threshold=0.9):
    faq_data_total = _create_dict_with_faqs(filenames)
    
    # get SBERT embeddings for the question and faq_question
    question_embed = model.encode(question)
    faq_questions = list(faq_data_total.keys())
    faq_questions_embed = model.encode(faq_questions)
    
    # Compute cosine similarity score between query and all FAQ question embeddings
    scores = cosine_similarity([question_embed], faq_questions_embed)[0].tolist()

    # Combine FAQ questions & scores
    faq_score_pairs = list(zip(faq_questions, scores))

    # Sort by decreasing score
    faq_score_pairs = sorted(faq_score_pairs, key=lambda x: x[1], reverse=True)
    highest_ranked_question, highest_score = faq_score_pairs[0]
    print(highest_score)
    
    # TODO: Maybe fine-tune the threshold
    if highest_score >= threshold:
        answer = faq_data_total[highest_ranked_question]
        print("QUESTION: ", highest_ranked_question)
        return answer
    else:
        return "No FAQ found"

app = App(token=BOT_TOKEN)

@app.event("message")
def mention_handler(body: dict, say: Callable):
    message = body["event"]["text"]
    ans = check_for_similar_qa(message, filenames, threshold=0.9)
    if ans == "No FAQ found":
        res = pipe.run(query=message, params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
        ans = res["answers"][0].answer
        
    say(ans)    

if __name__ == "__main__":
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()

