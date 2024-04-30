import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import SlackToolkit


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks, persist_dir='./chroma.db'):    
    embeddings = OpenAIEmbeddings(
        model= 'text-embedding-3-small',
        dimensions = 1536
    )    
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vector_store

# ask question from vector store and send it to slack
def ask_question(vector_store, q):    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    memory =ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    system_template = r'''
    Use the following pieces of context to answer the user's question.
    If you don't know the answer in the provided context, just respond "Data Not Available"
    -------------
    Context:```{context}```
    '''

    user_template = '''
    Question: ```{question}```
    Chat History: ```{chat_history}```
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    # Chain type stuff means use all of the text from the documents.
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type='stuff',
        memory=memory,
        combine_docs_chain_kwargs={'prompt': qa_prompt},
        verbose=False
    )

    result = crc.invoke({'question': q})
    json = {
        'question': result['question'],
        'answer': result['answer']
    }
    return json

def send_message_to_slack(message):
     # slack 
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()

    llm2 = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(
        tools=toolkit.get_tools(),
        llm=llm2,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    agent_executor.invoke({
        "input": f"Send {message} in the #gen-ai channel. Note use `channel` as key of channel id, and `message` as key of content to sent in the channel."
    })
    


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']



# App Entry point
if __name__ == "__main__":
    import os

    # You can authenticate using this method / you can create a text input
    # and the user will paste the API key.
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # Here we are adding image and sub-header
    st.image('logo-zania.png')
    st.divider()
    st.header('Chat with document')

    # We will organize some important widgets in a left panel sidebar with that sidebar.
    #
    # This way I save space and the user can focus on the main page content.
    with st.sidebar:
        # 1. Widget - Input Open API Key
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # 2 Widget - Input Slack Token
        # text_input for the Slack Token  (alternative to python-dotenv and .env)
        slack_token = st.text_input('Slack API Token:', type='password')
        if slack_token:
            os.environ['SLACK_USER_TOKEN'] = slack_token

        # 2. Widget - File uploader
        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # 3. Optionals Widget some advanced settings - `chunk size` and `k`
        # chunk size number widget
        #chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        chunk_size = 512

        # k number input widget
        #k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        k = 3

        # add data button widget
        add_data = st.button('Load Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            #st.write(f'k: {k}')
            #answer = ask_and_get_answer(vector_store, q, k)
            answer = ask_question(vector_store, q)

            # text area widget for the LLM answer
            st.text_area('Answer: ', value=answer)

            st.divider()

            # st.title('Sending message to slack .........')
            # send_message_to_slack(answer)
            # text area widget for the post answer into slack
            # st.title('Successfully pushed message to slack.')
            

            # # if there's no chat history in the session state, create it
            # if 'history' not in st.session_state:
            #     st.session_state.history = ''

            # # the current question and answer
            # value = f'Q: {q} \nA: {answer}'

            # st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            # h = st.session_state.history

            # # text area widget for the chat history
            # st.text_area(label='Chat History', value=h, key='history', height=400)