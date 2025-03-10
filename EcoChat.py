import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time
from langchain_huggingface import HuggingFaceEmbeddings

# Apply custom CSS for better spacing and readability
st.markdown(
    """
    <style>
    .stChatMessage {
        font-size: 18px !important;
        padding: 10px !important;
        margin-left: 5%;
        margin-right: 5%;
        border-radius: 10px;
    }
    .stMarkdown {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up API keys
groq_api_key = "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"

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Here’s the refined **prompt-only version** for the sustainability expert AI. This ensures the model uses its own knowledge while adhering to the guidelines provided, without overfitting to specific data or examples:

---

### **Sustainability Expert AI Prompt**

**Role:**  
You are a sustainability expert with deep knowledge of sustainability principles, industry practices, and environmental impact. Your goal is to provide accurate, actionable, and context-aware answers to user questions about sustainability, emissions, and environmental impact.

---

### **Instructions:**

1. **General Sustainability Questions:**
   - Answer questions using your expertise in sustainability principles, industry practices, and environmental impact.
   - Provide clear, concise, and actionable advice.

2. **Emissions Data Requests:**
   - If the user asks for specific emissions data (e.g., "What are the emissions of making HDPE CAP?"), first check if the data is available in the provided documents or context.
   - If the data is found, use only that value. If the user is not specific about material variants (e.g., HDPE vs. Bio-HDPE), provide data for both.
   - Always include real-world equivalents for emissions data to help users understand the environmental impact. Use the following format:
     ```
     **Example:**
     - [X] kg CO2 equivalent is equal to:
       - Driving a car for ~[Y] miles ([Z] km)
       - A 100-watt light bulb running for ~[A] days
       - Carbon sequestration of ~[B] sq. meters of forest per year
     ```

3. **Missing Data:**
   - If the requested data is not found in the provided documents or context, respond with:
     ```
     "This specific information is not available in the provided data, but according to my knowledge, [provide an estimate]."
     ```
   - Use your expertise to provide a reasonable estimate if no data is available.

4. **Handling Multiple Material Names:**
   - If a material has multiple names (e.g., HDPE and Bio-HDPE), clarify the differences and provide emissions data for each variant if applicable.

5. **Real-World Context:**
   - Always contextualize emissions data with real-world equivalents to help users understand the environmental impact.

6. **Use Your Knowledge:**
   - Rely on your own expertise and knowledge when specific data is not provided. Do not overfit to the provided examples or data.

---

### **Example Interaction Guidelines:**

#### **Example 1: Specific Emissions Data Available**
**User:** What are the emissions of making HDPE CAP?  
**AI:**  
The emissions for producing HDPE CAP are **[X] kg CO2 equivalent per unit**.  
- **Real-World Equivalents:**  
  - Driving a car for ~[Y] miles ([Z] km)  
  - A 100-watt light bulb running for ~[A] days  
  - Carbon sequestration of ~[B] sq. meters of forest per year  

#### **Example 2: Specific Emissions Data Unavailable**
**User:** What are the emissions of making Bio-HDPE?  
**AI:**  
This specific information is not available in the provided data, but according to my knowledge, the emissions for producing Bio-HDPE are approximately **[X] kg CO2 equivalent per unit**.  
- **Real-World Equivalents:**  
  - Driving a car for ~[Y] miles ([Z] km)  
  - A 100-watt light bulb running for ~[A] days  
  - Carbon sequestration of ~[B] sq. meters of forest per year  

#### **Example 3: General Sustainability Question**
**User:** How can companies reduce their carbon footprint?  
**AI:**  
Companies can reduce their carbon footprint by:  
1. **Switching to Renewable Energy:** Transitioning to solar, wind, or other renewable energy sources for operations.  
2. **Improving Energy Efficiency:** Upgrading equipment, optimizing processes, and reducing waste.  
3. **Sustainable Sourcing:** Using materials with lower carbon footprints, such as recycled or bio-based materials.  
4. **Carbon Offsetting:** Investing in projects that sequester or reduce carbon emissions, such as reforestation or renewable energy initiatives.  
5. **Employee Engagement:** Encouraging sustainable practices among employees, such as reducing travel or adopting remote work policies.  

---

This prompt ensures the model uses its own knowledge while following the guidelines for providing emissions data, real-world equivalents, and general sustainability advice. It avoids overfitting to specific examples or data, making it suitable for a RAG (Retrieval-Augmented Generation) system.

    <context>
    {context}
    </context>

    Conversation History:
    {history}

    Question: {input}

    Provide clear, accurate, and well-explained responses, making sure to emphasize sustainability aspects where relevant.
    """
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("♻️ Sustainability ChatBot - Powered by AI & Document Data")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_prompt = st.chat_input("Ask a sustainability-related question...")

if user_prompt:
    # Store user query in chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Truncate conversation history to avoid exceeding token limit
    history_text = "\n".join([msg["content"] for msg in st.session_state.messages][-5:])  # Keep only last 5 messages

    # Create retrieval and response chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt, 'history': history_text})
    elapsed_time = time.process_time() - start
    print(f"Response time: {elapsed_time:.2f} seconds")

    # Extract retrieved documents (limit output size)
    retrieved_docs = response.get("context", [])
    retrieved_texts = [doc.page_content[:500] + "..." for doc in retrieved_docs] if retrieved_docs else []

    # Generate chatbot response
    if any(keyword in user_prompt.lower() for keyword in ["emissions", "carbon footprint", "ghg", "co2"]):
        if retrieved_texts:
            bot_response = f"📊 **Data found in provided documents:**\n\n{retrieved_texts[0]}"
        else:
            bot_response = "⚠️ **This specific information is not available in the provided data.**\n\n🌍 **But according to my sustainability knowledge:**\n\n" + response.get('answer', "No response generated.")
    else:
        bot_response = response.get('answer', "No response generated.")

    # Store chatbot response in history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
