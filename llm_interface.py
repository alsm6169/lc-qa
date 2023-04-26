"""
This script provides functions for generating summaries and answering questions
from PDF documents using OpenAI's language model.
"""
import inspect
import tiktoken
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


def get_est_cost(model_name, split_docs):
    """
    Calculates the estimated cost of embedding the given split documents using the specified OpenAI language model.
   Args:
     model_name (str): The name of the OpenAI language model to use.
     split_docs (list of str): A list of strings representing the split documents.
   Returns:
     total_word_count (int): The total word count of the split documents.
     total_token_count (int): The total token count of the split documents.
     est_cost (float): The estimated cost of embedding the split documents using the specified OpenAI language model.
    """
    # the latest prices are here: https://openai.com/pricing#language-models
    price_map = {'gpt-4': 0.03, 'gpt-3.5-turbo': 0.002}
    model_price = price_map[model_name]
    enc = tiktoken.encoding_for_model(model_name)

    total_word_count = sum(len(pages.page_content.split()) for pages in split_docs)
    total_token_count = sum(len(enc.encode(pages.page_content)) for pages in split_docs)
    est_cost = total_token_count * model_price / 1000

    print(f'Total word count: {total_word_count}')
    print(f'Estimated tokens: {total_token_count}')
    print(f'Estimated cost of embedding: ${est_cost}')
    return total_word_count, total_token_count, est_cost


def get_doc_pages(doc_path_name):
    """
   Loads a PDF document and returns an array of documents, where each document contains the page content and metadata with page number.
   Args:
     doc_path_name (str): The path and name of the PDF document to load.
   Returns:
     pages (list of objects): A list of objects representing the pages of the PDF document.
    """
    # Load PDF using pypdf into array of documents,
    # where each document contains the page content and metadata with page number.
    # An advantage of this approach is that documents can be retrieved with page numbers.
    loader = PyPDFLoader(doc_path_name)

    pages = loader.load_and_split()
    # print(len(pages), type(pages[0]), pages[3])
    return pages


def get_summary_wo_prompt(llm, pages, chain_type):
    """
    Generates a summary of the given pages using the specified language model and summarization chain.
   Args:
     llm (OpenAI): The OpenAI language model to use for summarization.
     pages (list of objects): A list of objects representing the pages to summarize.
     chain_type (str): The type of summarization chain to use.
   Returns:
     summary_wo_prompt (str): The generated summary without a prompt.
    """
    chain_obj = load_summarize_chain(llm, chain_type=chain_type)
    # print(chain_obj.llm_chain.prompt.template)
    # print(chain_obj.combine_document_chain.llm_chain.prompt.template)
    summary_wo_prompt = chain_obj.run(pages)
    return summary_wo_prompt


def get_summary_with_prompt(llm, pages, chain_type):
    """
       Generates a summary of the given pages using the specified language model and summarization chain with a prompt.
   Args:
     llm (OpenAI): The OpenAI language model to use for summarization.
     pages (list of objects): A list of objects representing the pages to summarize.
     chain_type (str): The type of summarization chain to use.
   Returns:
     summary_with_prompt (str): The generated summary with a prompt.
    """
    prompt_template = """Write a concise summary of the following:

    {text}

    CONCISE SUMMARY:"""

    # print('load_summarize_chain params: ', inspect.signature(load_summarize_chain))
    prompt_instance = PromptTemplate(template=prompt_template, input_variables=['text'])
    chain_obj = load_summarize_chain(llm, chain_type=chain_type, return_intermediate_steps=True,
                                     map_prompt=prompt_instance, combine_prompt=prompt_instance)
    summary_with_prompt = chain_obj({'input_documents': pages}, return_only_outputs=True)
    return summary_with_prompt['output_text']


def get_summary(doc_path_name, model_name='gpt-3.5-turbo', chain_type='map_reduce'):
    """
    Generates a summary of the PDF document at the specified path and name using the specified language model and summarization chain.
   Args:
     doc_path_name (str): The path and name of the PDF document to summarize.
     model_name (str): The name of the OpenAI language model to use for summarization. Defaults to 'gpt-3.5-turbo'.
     chain_type (str): The type of summarization chain to use. Defaults to 'map_reduce'.
   Returns:
     summary_wo_prompt (str): The generated summary without a prompt.
    """
    pages = get_doc_pages(doc_path_name)
    # return pages[1].page_content
    llm = OpenAI(temperature=0)
    # llm = OpenAI(model_name=model_name, temperature=0)
    summary_wo_prompt = get_summary_wo_prompt(llm, pages, chain_type)
    print('summary_wo_prompt: ', summary_wo_prompt)
    return summary_wo_prompt

    # summary_with_prompt = get_summary_with_prompt(llm, pages, chain_type)
    # print('summary_with_prompt: ', summary_with_prompt)
    #
    # return summary_with_prompt


def answer_the_question(doc_path_name, question, model_name='gpt-3.5-turbo', chain_type='map_reduce', num_matches=3):
    """
    Generates an answer to the given question from the PDF document at the specified path and name using the specified language model and summarization chain.
   Args:
     doc_path_name (str): The path and name of the PDF document to search for an answer.
     question (str): The question to answer.
     model_name (str): The name of the OpenAI language model to use for summarization. Defaults to 'gpt-3.5-turbo'.
     chain_type (str): The type of summarization chain to use. Defaults to 'map_reduce'.
     num_matches (int): The number of most relevant documents to search for an answer. Defaults to 3.
   Returns:
     answer (str): The generated answer to the
    """
    chunk_size_limit = 1000
    max_chunk_overlap = 50
    embeddings_dir = './'
    pages = get_doc_pages(doc_path_name)
    # return pages[3].page_content
    embeddings = OpenAIEmbeddings()
    try:
        # load the embeddings stored in FAISS vector_db
        vector_db = FAISS.load_local(embeddings_dir, embeddings)
    except Exception as e:
        print('No embeddings found: (will create now) ')  # , e)
        get_est_cost(model_name, pages)
        vector_db = FAISS.from_documents(pages, embeddings)
        # Save the embeddings vector store files `./index.faiss` and `./index.pkl`
        vector_db.save_local(embeddings_dir)

    relevant_docs = vector_db.similarity_search(question, num_matches=num_matches)
    # for doc in relevant_docs:
    #     print(str(doc.metadata["page"]) + ":", doc.page_content)

    question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    {context}
    Question: {question}
    Relevant text:"""
    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    llm = OpenAI(temperature=0)
    chain_obj = load_qa_with_sources_chain(llm, chain_type="map_reduce", return_intermediate_steps=True,
                                           question_prompt=question_prompt, combine_prompt=combine_prompt)
    ans = chain_obj({'input_documents': relevant_docs, 'question': question}, return_only_outputs=True)
    # print('question:', question, '\nanswer: ', ans)
    return ans['output_text']


if __name__ == '__main__':
    model = 'gpt-3.5-turbo'
    chain = 'map_reduce'
    file = 'documents/How to Apply for a Job.pdf'

    summary = get_summary(file, model, chain)
    print('summary: ', summary)

    q1 = 'how to write cold email?'
    a1 = answer_the_question(file, q1)
    print('q1: ', q1, '\na1: ', a1)
