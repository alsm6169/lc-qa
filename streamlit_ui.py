import streamlit as st
import os

from llm_interface import get_summary, answer_the_question

# Create an empty list to store the question-answer history
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

if 'q_input_box' not in st.session_state:
    st.session_state.q_input_box = ''

if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

if 'files_2_upload' not in st.session_state:
    st.session_state.files_2_upload = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'disabled' not in st.session_state:
    st.session_state.disabled = False


def write_files_info(out_col):
    # print('inside write_files_info')
    for i in range(len(st.session_state.files_2_upload)):
        out_col.write('File - ' + st.session_state.files_2_upload[i]['Name'])
        out_col.write('Type - ' + st.session_state.files_2_upload[i]['Type'])
        out_col.write('Size - ' + st.session_state.files_2_upload[i]['Size'])
        out_col.write('Summary - ' + st.session_state.files_2_upload[i]['Summary'])
        out_col.divider()


# Create a function to handle user input and server replies
def handle_question(question):
    # print('inside handle_question: ', question)
    # generate a response
    file = st.session_state.uploaded_files[0]
    response = answer_the_question(file, question)
    # response = str(random.randint(1, 1000))

    # Add the question and answer to the history
    st.session_state.qa_history.append((question, response))

    # Return the response to display to the user
    return response


def save_files_get_summary():
    print('inside save_files_get_summary: ', st.session_state.file_upload_widget)
    doc_dir = 'documents'
    if len(st.session_state.file_upload_widget) > 0:
        for file in st.session_state.file_upload_widget:
            file_path = os.path.join(doc_dir, file.name)
            st.session_state.uploaded_files.append(file_path)
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            summary = get_summary(file_path)
            st.session_state.files_2_upload.append({'Name': file.name,
                                                    'Type': file.type,
                                                    'Size': str(file.size),
                                                    'Summary': summary})

        st.session_state.disabled = True
        return True
    else:
        st.session_state.disabled = False
        return False


def clear_text_box():
    st.session_state.current_question = st.session_state.q_input_box
    st.session_state.q_input_box = ''


# Set up the Streamlit app
def main():

    # sidebar creation and handling
    st.sidebar.title('File Upload and Processing')
    with st.sidebar.form(key='sidebar_form'):
        # Allow the user to upload a files
        files_2_upload = st.file_uploader('Upload files',
                                          type=['pdf', 'txt'],
                                          key='file_upload_widget',
                                          accept_multiple_files=True,
                                          disabled=st.session_state.disabled)
        # print('main files_2_upload: ', files_2_upload)
        # If a files was uploaded, display its contents
        submit_btn = st.form_submit_button('Upload Files',
                                           on_click=save_files_get_summary,
                                           disabled=st.session_state.disabled)
                                    #        args = (files_2_upload,),
        if submit_btn:
            st.sidebar.write('No more upload possible')

    # main screen creation and handling
    col1, col2 = st.columns(2)

    col1.subheader('Summary')
    write_files_info(col1)

    # Allow the user to input questions
    col2.subheader('Question')
    col2.text_input('Ask a question', key='q_input_box', on_change=clear_text_box)
    question = st.session_state.current_question

    # If the user asks a question, display the server's response and update the history
    col2.subheader('Answer')
    if len(question) > 0:
        response = handle_question(question)
        col2.write('Question: ' + question)
        col2.write('Reply: ' + response)
        # reset the question since it has been handled
        question = ''
        st.session_state.current_question = ''

    # below the divider is the chat history
    st.divider()
    st.subheader('Session History')
    # Display the question-answer history, with the most recent items at the top
    with st.expander('Question-Answer History'):
        for qa_pair in reversed(st.session_state.qa_history):
            st.write(f'Question: {qa_pair[0]}')
            st.write(f'Answer: {qa_pair[1]}')
            st.write('')


if __name__ == "__main__":
    main()
