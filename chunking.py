from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_chunk(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # size of each chunk
        chunk_overlap=50,     # overlap between chunks
        separators=[
            "\n\n",  # paragraph
            "\n",    # line
            ".",     # sentence 
            " ",     # word
            ""       # character
        ]
    )
 
    return splitter.split_text(text) 